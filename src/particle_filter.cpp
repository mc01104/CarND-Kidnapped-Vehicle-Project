/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

// declare a random engine to be used across multiple and various method calls
void ParticleFilter::init(double x, double y, double theta, double std[])
{
  num_particles = 101;

  // define normal distributions for sensor noise
  std::normal_distribution<double> x_noise(0, std[0]);
  std::normal_distribution<double> y_noise(0, std[1]);
  std::normal_distribution<double> theta_noise(0, std[2]);

  Particle p;

  // init particles
  for (int i = 0; i < num_particles; ++i)
  {
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;

    p.weight = 1.0;

    // add noise
    p.x += x_noise(gen);
    p.y += y_noise(gen);
    p.theta += theta_noise(gen);

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  // define normal distributions for sensor noise
  std::normal_distribution<double> x_noise(0, std_pos[0]);
  std::normal_distribution<double> y_noise(0, std_pos[1]);
  std::normal_distribution<double> theta_noise(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i)
  {

    // calculate new state
    if (fabs(yaw_rate) < 0.00001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);

      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));

      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));

      particles[i].theta += yaw_rate * delta_t;
    }

    // add noise
    particles[i].x += x_noise(gen);
    particles[i].y += y_noise(gen);
    particles[i].theta += theta_noise(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  for (int i = 0; i < observations.size(); ++i)
  {

    double min_dist = std::numeric_limits<double>::max();

    int id_m = -1;

    for (int j = 0; j < predicted.size(); ++j)
    {

      auto cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

      if (cur_dist < min_dist)
      {
        min_dist = cur_dist;
        id_m = predicted[j].id;
      }
    }

    observations[i].id = id_m;
  }
}

// needs refactoring
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  for (int i = 0; i < num_particles; ++i)
  {

    auto p_x = particles[i].x;
    auto p_y = particles[i].y;
    auto p_theta = particles[i].theta;
    std::vector<LandmarkObs> landmarks_within_range;

    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
    {

      auto lm_x = map_landmarks.landmark_list[j].x_f;
      auto lm_y = map_landmarks.landmark_list[j].y_f;
      auto lm_id = map_landmarks.landmark_list[j].id_i;

      //maybe change to check the distance
      if (fabs(lm_x - p_x) <= sensor_range && fabs(lm_y - p_y) <= sensor_range)
      {
        landmarks_within_range.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }

    std::vector<LandmarkObs> transformed_os;
    for (int j = 0; j < observations.size(); ++j)
    {
      auto t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
      auto t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
      transformed_os.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }

    this->dataAssociation(landmarks_within_range, transformed_os);

    particles[i].weight = 1.0;

    for (int j = 0; j < transformed_os.size(); ++j)
    {

      double o_x, o_y, pr_x, pr_y;
      o_x = transformed_os[j].x;
      o_y = transformed_os[j].y;

      int associated_prediction = transformed_os[j].id;

      for (int k = 0; k < landmarks_within_range.size(); ++k)
      {
        if (landmarks_within_range[k].id == associated_prediction)
        {
          pr_x = landmarks_within_range[k].x;
          pr_y = landmarks_within_range[k].y;
        }
      }

      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = exp(-(pow(pr_x - o_x, 2) / (2.0 * pow(s_x, 2)) + (pow(pr_y - o_y, 2) / (2.0 * pow(s_y, 2)))));
      obs_w *= 1.0 / (2.0 * M_PI * s_x * s_y);
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  ::std::vector<Particle> new_particles;

  ::std::vector<double> current_weights;
  for (int i = 0; i < num_particles; i++)
  {
    current_weights.push_back(particles[i].weight);
  }
  std::uniform_int_distribution<int> uniintdist(0, num_particles - 1);
  auto index = uniintdist(gen);

  double max_weight = *max_element(current_weights.begin(), current_weights.end());

  std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  for (int i = 0; i < num_particles; ++i)
  {
    beta += unirealdist(gen) * 2.0;
    while (beta > current_weights[index])
    {
      beta -= current_weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  std::vector<int> v = best.associations;
  std::stringstream ss;
  std::copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  std::vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}