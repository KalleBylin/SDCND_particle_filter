/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * 
 * Completed on: May 15th, 2019
 * By: Kalle Bylin
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  weights.resize(num_particles);
  particles.resize(num_particles);
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i=0; i<num_particles; ++i) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
    weights[i] = 1.0;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for (int i=0; i<num_particles; ++i) {
    // Watch out for very small numbers of yaw rate in division
    if(fabs(yaw_rate) < 0.0001) {
      particles[i].x = velocity*delta_t*cos(particles[i].theta);
      particles[i].y = velocity*delta_t*sin(particles[i].theta);
    } else {
      // Noise is added with '+ dist_x(gen)'
      particles[i].x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta)) + dist_x(gen);
      // Noise is added with '+ dist_y(gen)'
      particles[i].y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t)) + dist_y(gen);
    particles[i].theta = particles[i].theta + yaw_rate*delta_t + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (unsigned int i=0; i<observations.size(); ++i) {
    double closest_dist = std::numeric_limits<double>::max();
    for (unsigned int j=0; j<predicted.size(); ++j) {
      double distan = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      
      if (distan <= closest_dist) {
        closest_dist = distan;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  auto covar_x = std_landmark[0] * std_landmark[0];
  auto covar_y = std_landmark[1] * std_landmark[1];
  auto gaussian_norm = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
  
  for (int i=0; i<num_particles; ++i) {
    vector<LandmarkObs> pred_landmarks;
    
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); ++j) {
      float landm_x = map_landmarks.landmark_list[j].x_f;
      float landm_y = map_landmarks.landmark_list[j].y_f;
      int landm_id = map_landmarks.landmark_list[j].id_i;
      
      if(dist(landm_x, landm_y, particles[i].x, particles[i].y) <= sensor_range) {
        pred_landmarks.push_back(LandmarkObs{landm_id, landm_x, landm_y});
	  }
    }
    
    if(pred_landmarks.size() == 0) {
      particles[i].weight = 0;
      weights[i] = 0;
    } else {
      vector<LandmarkObs> trans_obs;
      for (unsigned int k=0; k<observations.size(); ++k) {
        double cos_theta = cos(particles[i].theta);
        double sin_theta = sin(particles[i].theta);
        trans_obs[k].x = particles[i].x + cos_theta * observations[k].x - sin_theta * observations[k].y;
        trans_obs[k].y = particles[i].y + sin_theta * observations[k].x + cos_theta * observations[k].y;
      }
      
      dataAssociation(pred_landmarks, trans_obs);
      
      double prob =1.0;
      for (unsigned int l=0; l<trans_obs.size(); ++l) {
        double dx = trans_obs[l].x - pred_landmarks[trans_obs[l].id].x;
        double dy = trans_obs[l].y - pred_landmarks[trans_obs[l].id].y;
        prob *= exp(-(dx*dx / (2*covar_x) + dy*dy / (2*covar_y))) / gaussian_norm;
      }
      particles[i].weight = prob;
      weights[i] = prob;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  double maxW = *std::max_element(weights.begin(), weights.end());
  std::uniform_real_distribution<double> dist_double(0.0, maxW);
  std::uniform_int_distribution<int> dist_int(0, num_particles - 1);
  
  int idx = dist_int(gen);
  double beta = 0.0;
  vector<Particle> new_particles;
  for(int i=0; i<num_particles; ++i) {
	beta += dist_double(gen) * 2.0;
	while(beta > weights[idx]) {
		beta -= weights[idx];
		idx = (idx + 1) % num_particles;
	}
	new_particles.push_back(particles[idx]);
  }
  particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
