#include <mutex>
#include <thread>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <HAL/Camera/CameraDevice.h>
#include <HAL/Posys/PosysDevice.h>
#include <matchbox/matchbox.h>
#include <vulcan/vulcan.h>

#include <fstream>

DEFINE_string(cam, "", "HAL camera uri");
DEFINE_string(posys, "vicon://tracker:[darpa]", "HAL posys uri");
DEFINE_int32(max_disp, 128, "max disparity");
DEFINE_int32(degree, 3, "aggregate cost degree");
DEFINE_bool(check, true, "perform disparity check");
DEFINE_bool(check_both, false, "perform disparity check on both frames");
DEFINE_bool(filter, true, "perform median filter before disparity check");
DEFINE_double(uniqueness, 0.95, "uniqueness threshold for disparity computer");
DEFINE_int32(main_blocks, 260096, "main block count");
DEFINE_int32(excess_blocks, 8192, "excess block count");
DEFINE_double(voxel_length, 0.02, "voxel edge length");
DEFINE_double(truncation_length, 0.08, "volume truncation length");

std::mutex pose_mutex;
vulcan::Transform last_transform;
bool transform_updated;

std::ofstream pose_stream;

inline int GetDirections()
{
  switch (FLAGS_degree)
  {
    case 0: return matchbox::Aggregator::DIR_NONE;
    case 1: return matchbox::Aggregator::DIR_HORIZONTAL;
    case 2: return matchbox::Aggregator::DIR_HORIZONTAL_VERTICAL;
    case 3: return matchbox::Aggregator::DIR_ALL;
  }

  MATCHBOX_THROW("invalid degree");
}

matchbox::DisparityChecker::Mode GetCheckerMode()
{
  return FLAGS_check_both ?
      matchbox::DisparityChecker::MODE_CHECK_BOTH :
      matchbox::DisparityChecker::MODE_CHECK_LEFT;
}

void Copy(const matchbox::DepthImage& src, vulcan::Image& dst)
{
  dst.Resize(src.GetWidth(), src.GetHeight());
  const cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
  cudaMemcpy(dst.GetData(), src.GetData(), src.GetBytes(), kind);
}

void PoseCallback(hal::PoseMsg& pose)
{
  std::lock_guard<std::mutex> lock(pose_mutex);

  // OptiTrack (darpa object):
  // +X: forward
  // +Y: up
  // +Z: right


  // Vulcan:
  // +X: right
  // +Y: down
  // +Z: foward

  vulcan::Vector4f r;

  r[0] = pose.pose().data(6);
  r[1] = pose.pose().data(3);
  r[2] = pose.pose().data(4);
  r[3] = pose.pose().data(5);

  // r[0] = pose.pose().data(3);
  // r[1] = pose.pose().data(4);
  // r[2] = pose.pose().data(5);
  // r[3] = pose.pose().data(6);

  vulcan::Transform R = vulcan::Transform::Rotate(r);
  const vulcan::Matrix4f& tracker_matrix = R.GetMatrix();
  vulcan::Matrix3f vulcan_matrix;

  vulcan_matrix(0, 0) = -tracker_matrix(0, 1);
  vulcan_matrix(1, 0) = -tracker_matrix(1, 1);
  vulcan_matrix(2, 0) = -tracker_matrix(2, 1);

  vulcan_matrix(0, 1) = -tracker_matrix(0, 2);
  vulcan_matrix(1, 1) = -tracker_matrix(1, 2);
  vulcan_matrix(2, 1) = -tracker_matrix(2, 2);

  vulcan_matrix(0, 2) =  tracker_matrix(0, 0);
  vulcan_matrix(1, 2) =  tracker_matrix(1, 0);
  vulcan_matrix(2, 2) =  tracker_matrix(2, 0);

  // vulcan_matrix(0, 0) =  tracker_matrix(0, 0);
  // vulcan_matrix(1, 0) =  tracker_matrix(1, 0);
  // vulcan_matrix(2, 0) =  tracker_matrix(2, 0);

  // vulcan_matrix(0, 1) =  tracker_matrix(0, 1);
  // vulcan_matrix(1, 1) =  tracker_matrix(1, 1);
  // vulcan_matrix(2, 1) =  tracker_matrix(2, 1);

  // vulcan_matrix(0, 2) =  tracker_matrix(0, 2);
  // vulcan_matrix(1, 2) =  tracker_matrix(1, 2);
  // vulcan_matrix(2, 2) =  tracker_matrix(2, 2);

  R = vulcan::Transform::Rotate(vulcan_matrix);

  vulcan::Vector3f t;
  t[0] =  pose.pose().data(0);
  t[1] =  pose.pose().data(1);
  t[2] =  pose.pose().data(2);

  vulcan::Transform T = vulcan::Transform::Translate(t);

  last_transform = T * R;
  transform_updated = true;

  const vulcan::Matrix4f& M = last_transform.GetMatrix();

  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      pose_stream << M(j, i);
      pose_stream << ((i * 4 + j < 15) ? ' ' : '\n');
    }
  }

  // for (int i = 0; i < 6; ++i)
  // {
  //   pose_stream << pose.pose().data(i) << " ";
  // }

  // pose_stream << pose.pose().data(6) << std::endl;
}

int main(int argc, char** argv)
{
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO) << "Starting up...";

  pose_stream.open("poses.txt");

  hal::Camera camera(FLAGS_cam);
  std::vector<cv::Mat> images(2);

  hal::Posys posys(FLAGS_posys);
  posys.RegisterPosysDataCallback(PoseCallback);
  transform_updated = false;

  std::shared_ptr<matchbox::Image> left_result;
  std::shared_ptr<matchbox::Image> right_result;
  std::shared_ptr<matchbox::Image> left_image;
  std::shared_ptr<matchbox::Image> right_image;
  std::shared_ptr<matchbox::FeatureMap> left_features;
  std::shared_ptr<matchbox::FeatureMap> right_features;
  std::shared_ptr<matchbox::MatchingCost> matching_cost;
  std::shared_ptr<matchbox::AggregateCost> aggregate_cost;
  std::shared_ptr<matchbox::Image> left_disparities;
  std::shared_ptr<matchbox::Image> right_disparities;
  std::shared_ptr<matchbox::DepthImage> depth;

  left_image = std::make_shared<matchbox::Image>();
  right_image = std::make_shared<matchbox::Image>();
  left_features = std::make_shared<matchbox::FeatureMap>();
  right_features = std::make_shared<matchbox::FeatureMap>();
  matching_cost = std::make_shared<matchbox::MatchingCost>();
  aggregate_cost = std::make_shared<matchbox::AggregateCost>();
  left_disparities = std::make_shared<matchbox::Image>();
  right_disparities = std::make_shared<matchbox::Image>();
  depth = std::make_shared<matchbox::DepthImage>();

  // TODO: use flags
  matchbox::Calibration calibration;
  calibration.baseline = 0.1205;
  calibration.focal_length = 258.25;
  calibration.left_center_point = 325.8055;
  calibration.right_center_point = 326.4072;

  std::shared_ptr<vulcan::Volume> volume;
  const int main_blocks = FLAGS_main_blocks;
  const int excess_blocks = FLAGS_excess_blocks;
  volume = std::make_shared<vulcan::Volume>(main_blocks, excess_blocks);
  volume->SetTruncationLength(FLAGS_truncation_length);
  volume->SetVoxelLength(FLAGS_voxel_length);

  vulcan::Integrator integrator(volume);

  vulcan::Frame frame;
  frame.transform = vulcan::Transform::Translate(0, 0, 0);
  frame.projection.SetFocalLength(258.2812, 293.2650); // TODO: use flags
  frame.projection.SetCenterPoint(325.8055, 268.8094); // TODO: use flags
  frame.depth_image = std::make_shared<vulcan::Image>();

  vulcan::Mesh mesh;
  vulcan::Extractor extractor(volume);
  vulcan::Exporter exporter("output.ply");

  // while (true)
  for (int i = 0; i < 18; ++i)
  {
    if (!transform_updated)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    LOG(INFO) << "Capturing pose...";
    {
      std::lock_guard<std::mutex> lock(pose_mutex);
      frame.transform = last_transform;
    }

    LOG(INFO) << "Capturing images...";
    const bool capture_successful = camera.Capture(images);

    if (!capture_successful)
    {
      LOG(INFO) << "Failed to capture from camera";
      break;
    }

    LOG(INFO) << "Uploading images...";
    left_image->Load(images[0]);
    right_image->Load(images[1]);

    LOG(INFO) << "Extracting left features...";
    matchbox::FeatureExtractor left_extractor(left_image);
    left_extractor.Extract(*left_features);

    LOG(INFO) << "Extracting right features...";
    matchbox::FeatureExtractor right_extractor(right_image);
    right_extractor.Extract(*right_features);

    LOG(INFO) << "Computing matching cost...";
    matchbox::Matcher matcher(left_features, right_features);
    matcher.SetMaxDisparity(FLAGS_max_disp);
    matcher.Match(*matching_cost);

    LOG(INFO) << "Computing aggregate cost...";
    matchbox::Aggregator aggregator(matching_cost);
    aggregator.SetDirections(GetDirections());
    aggregator.Aggregate(*aggregate_cost);

    LOG(INFO) << "Computing left disparities...";
    matchbox::DisparityComputer left_computer(aggregate_cost);
    left_computer.SetUniqueness(FLAGS_uniqueness);
    left_computer.SetInverted(false);
    left_computer.Compute(*left_disparities);
    left_result = left_disparities;

    if (FLAGS_filter)
    {
      left_result = std::make_shared<matchbox::Image>();
      matchbox::MedianFilter left_filter(left_disparities);
      left_filter.Filter(*left_result);
    }

    if (FLAGS_check)
    {
      LOG(INFO) << "Computing right disparities...";
      matchbox::DisparityComputer right_computer(aggregate_cost);
      right_computer.SetUniqueness(FLAGS_uniqueness);
      right_computer.SetInverted(true);
      right_computer.Compute(*right_disparities);
      right_result = right_disparities;

      if (FLAGS_filter)
      {
        LOG(INFO) << "Filtering right disparities...";
        right_result = std::make_shared<matchbox::Image>();
        matchbox::MedianFilter right_filter(right_disparities);
        right_filter.Filter(*right_result);
      }

      LOG(INFO) << "Checking disparities...";
      matchbox::DisparityChecker checker(left_result, right_result);
      checker.SetMode(GetCheckerMode());
      checker.SetMaxDifference(1);
      checker.Check();
    }

    LOG(INFO) << "Converting disparities...";
    matchbox::DisparityConverter converter(left_result);
    converter.SetCalibration(calibration);
    converter.Convert(*depth);

    LOG(INFO) << "Copying depth image...";
    Copy(*depth, *frame.depth_image);

    LOG(INFO) << "Allocating voxel blocks...";
    volume->SetView(frame);

    LOG(INFO) << "Integrating depth image...";
    integrator.Integrate(frame);

    LOG(INFO) << "Extracting mesh...";
    extractor.Extract(mesh);

    LOG(INFO) << "Exporting mesh...";
    exporter.Export(mesh);

    frame.transform = frame.transform * vulcan::Transform::Translate(0, 0, 0.1);
  }

  pose_stream.close();

  LOG(INFO) << "Shutting down...";
  return 0;
}