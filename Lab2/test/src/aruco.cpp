#include <string>
#include <algorithm>
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/aruco.hpp>
#include <catch2/catch.hpp>

SCENARIO("markers can be detected with ArUco", "[aruco]")
{
    GIVEN("an image with 6 markers")
    {
        std::string fileName = "test/data/singlemarkersoriginal.jpg";
        REQUIRE(std::filesystem::exists(fileName));

        cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);
        REQUIRE_FALSE(image.empty());

        WHEN("detecting markers")
        {
            cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f> > corners;
            cv::aruco::detectMarkers(image, dictionary, corners, ids);

            THEN("image contains 6 markers")
            {
                REQUIRE(ids.size() == 6);

                AND_THEN("markers have expected IDs")
                {
                    std::vector<int> expectedIDs = {23, 40, 62, 98, 124, 203};
                    std::vector<int> actualIDs = ids;
                    std::sort(actualIDs.begin(), actualIDs.end());
                    REQUIRE(actualIDs == expectedIDs);
                }

                REQUIRE(corners.size() == 6);
            }

            THEN("each marker has 4 corners")
            {
                for (const auto & x : corners)
                    REQUIRE(x.size() == 4);
            }
        }
    }
}

SCENARIO("markers cannot be detected with ArUco", "[aruco]")
{
    GIVEN("a meme image with no markers")
    {
        std::string fileName = "test/data/negativeAdrian.png";
        REQUIRE(std::filesystem::exists(fileName));

        cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);
        REQUIRE_FALSE(image.empty());

        WHEN("detecting markers")
        {
            cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f> > corners;
            cv::aruco::detectMarkers(image, dictionary, corners, ids);

            THEN("image contains no markers")
            {
                REQUIRE(ids.empty());
                REQUIRE(corners.empty());
            }
        }
    }
}