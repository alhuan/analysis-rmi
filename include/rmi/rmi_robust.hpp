#pragma once

#include <algorithm>
#include <vector>
#include <numeric>


namespace rmi {

/**
 * Struct to hold the approximated position and error bounds returned by the index.
 */
struct Approx {
  std::size_t pos; ///< The estimated position of the key.
  std::size_t lo;  ///< The lower bound of the search range.
  std::size_t hi;  ///< The upper bound of the search range.
};

/**
 * This is a reimplementation of a two-layer recursive model index (RMI) supporting a variety of (monotonic) models.
 * RMIs were invented by Kraska et al. (https://dl.acm.org/doi/epdf/10.1145/3183713.3196909).
 *
 * Note that this is the base class which does not provide error bounds.
 *
 * @tparam Key the type of the keys to be indexed
 * @tparam Layer1 the type of the model used in layer1
 * @tparam Layer2 the type of the models used in layer2
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiRobust
{
  using key_type = Key;
  using layer1_type = Layer1;
  using layer2_type = Layer2;

 protected:
  std::size_t n_keys_;      ///< The number of keys the index was built on.
  std::size_t layer2_size_; ///< The number of models in layer2.
  layer1_type l1_;          ///< The layer1 model.
  layer2_type *l2_;         ///< The array of layer2 models.
  float outliers_;          ///< The proportion of elements to disregard as outliers
  float avg_error_;

 public:
  /**
   * Default constructor.
   */
  RmiRobust() = default;

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
   * @param keys vector of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  RmiRobust(const std::vector<key_type> &keys, const std::size_t layer2_size, const float outliers)
      : RmiRobust(keys.begin(), keys.end(), layer2_size, outliers) { }

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
   * @param first, last iterators that define the range of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  template<typename RandomIt>
  RmiRobust(RandomIt first, RandomIt last, const std::size_t layer2_size, const float outliers)
      : n_keys_(std::distance(first, last))
      , layer2_size_(layer2_size)
      , outliers_(outliers)
      , avg_error_(0)
  {
    std::size_t outlier_size = (std::size_t) (outliers_ * n_keys_);
    // Train layer1.
    l1_ = layer1_type(first + outlier_size, last - outlier_size, outlier_size, static_cast<double>(layer2_size) / n_keys_); // train with compression

    // Train layer2.
    l2_ = new layer2_type[layer2_size];
    std::size_t segment_start = 0;
    std::size_t segment_id = 0;
    // Assign each key to its segment.
    for (std::size_t i = 0; i != n_keys_; ++i) {
      auto pos = first + i;
      std::size_t pred_segment_id = get_segment_id(*pos);
      // If a key is assigned to a new segment, all models must be trained up to the new segment.
      if (pred_segment_id > segment_id) {
        new (&l2_[segment_id]) layer2_type(first + segment_start, pos, segment_start);
        for (std::size_t j = segment_id + 1; j < pred_segment_id; ++j) {
          new (&l2_[j]) layer2_type(pos - 1, pos, i - 1); // train other models on last key in previous segment
        }
        segment_id = pred_segment_id;
        segment_start = i;
      }
    }
    // Train remaining models.
    new (&l2_[segment_id]) layer2_type(first + segment_start, last, segment_start);
    for (std::size_t j = segment_id + 1; j < layer2_size; ++j) {
      new (&l2_[j]) layer2_type(last - 1, last, n_keys_ - 1); // train remaining models on last key
    }
  }

  /**
   * Destructor.
   */
  ~RmiRobust() { delete[] l2_; }

  /**
   * Returns the id of the segment @p key belongs to.
   * @param key to get segment id for
   * @return segment id of the given key
   */
  std::size_t get_segment_id(const key_type key) const {
    return std::clamp<double>(l1_.predict(key), 0, layer2_size_ - 1);
  }

  /**
   * Returns a position estimate and search bounds for a given key.
   * @param key to search for
   * @return position estimate and search bounds
   */
  Approx search(const key_type key) const {
    auto segment_id = get_segment_id(key);
    std::size_t pred = std::clamp<double>(l2_[segment_id].predict(key), 0, n_keys_ - 1);
    return {pred, 0, n_keys_};
  }

  /**
   * Returns the number of keys the index was built on.
   * @return the number of keys the index was built on
   */
  std::size_t n_keys() const { return n_keys_; }

  /**
   * Returns the number of models in layer2.
   * @return the number of models in layer2
   */
  std::size_t layer2_size() const { return layer2_size_; }

  /**
   * Returns the size of the index in bytes.
   * @return index size in bytes
   */
  std::size_t size_in_bytes() {
    return l1_.size_in_bytes() + layer2_size_ * l2_[0].size_in_bytes() + sizeof(n_keys_) + sizeof(layer2_size_);
  }

  /**
   * Returns a representation of the number of segments within each "bin" of the data
   * @return vector with number of segments in each bin
   */
   std::vector<std::size_t> segments_per_bin(const std::vector<key_type> &keys, std::size_t num_bins) {
     std::vector<std::size_t> bin_segments;
     auto bin_size = keys.size() / num_bins;
     bin_segments.push_back(get_segment_id(keys[bin_size] - 1));
     for (std::size_t i = 2; i <= num_bins; i++) {
        bin_segments.push_back(get_segment_id(keys[bin_size * i] - 1) - bin_segments[bin_segments.size() - 1]);
     }
     return bin_segments;
   }

   float mean_error() {
       return avg_error_;
   }

   std::vector<std::size_t> keys_per_segment(const std::vector<key_type>&keys) {
       std::vector<std::size_t> segments;
       for (std::size_t i = 0; i < n_keys_; i++) {
           auto pred_segment = get_segment_id(keys[i]);
           while (segments.size() < pred_segment + 1) {
               segments.push_back(0);
           }
           ++segments[pred_segment];
       }
       return segments;
   }
};


/**
 * Recursive model index with global absolute bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiGAbsRobust : public RmiRobust<Key, Layer1, Layer2>
{
  using base_type = RmiRobust<Key, Layer1, Layer2>;
  using key_type = Key;
  using layer1_type = Layer1;
  using layer2_type = Layer2;

 protected:
  std::size_t error_; ///< The error bound of the layer2 models.

 public:
  /**
   * Default constructor.
   */
  RmiGAbsRobust() = default;

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
   * @param keys vector of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  RmiGAbsRobust(const std::vector<key_type> &keys, const std::size_t layer2_size, const float outliers)
      : RmiGAbsRobust(keys.begin(), keys.end(), layer2_size, outliers) { }

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
   * @param first, last iterators that define the range of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  template<typename RandomIt>
  RmiGAbsRobust(RandomIt first, RandomIt last, const std::size_t layer2_size, const float outliers) : base_type(first, last, layer2_size, outliers) {
    // Compute global absolute errror bounds.
    error_ = 0;
    for (std::size_t i = 0; i != base_type::n_keys_; ++i) {
      key_type key = *(first + i);
      std::size_t segment_id = base_type::get_segment_id(key);
      std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
      if (pred > i) { // overestimation
        error_ = std::max(error_, pred - i);
        base_type::avg_error_ =  i * base_type::avg_error_ / (i + 1) + static_cast<float>(pred - i) / (i + 1);
      } else { // underestimation
        error_ = std::max(error_, i - pred);
        base_type::avg_error_ =  i * base_type::avg_error_ / (i + 1) + static_cast<float>(i - pred) / (i + 1);
      }
    }
  }

  /**
   * Returns a position estimate and search bounds for a given key.
   * @param key to search for
   * @return position estimate and search bounds
   */
  Approx search(const key_type key) const {
    auto segment_id = base_type::get_segment_id(key);
    std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
    std::size_t lo = pred > error_ ? pred - error_ : 0;
    std::size_t hi = std::min(pred + error_ + 1, base_type::n_keys_);
    return {pred, lo, hi};
  }

  /**
   * Returns the size of the index in bytes.
   * @return index size in bytes
   */
  std::size_t size_in_bytes() { return base_type::size_in_bytes() + sizeof(error_); }
};


/**
 * Recursive model index with global individual bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiGIndRobust : public RmiRobust<Key, Layer1, Layer2>
{
  using base_type = RmiRobust<Key, Layer1, Layer2>;
  using key_type = Key;
  using layer1_type = Layer1;
  using layer2_type = Layer2;

 protected:
  std::size_t error_lo_; ///< The lower error bound of the layer2 models.
  std::size_t error_hi_; ///< The upper error bound of the layer2 models.

 public:
  /**
   * Default constructor.
   */
  RmiGIndRobust() = default;

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
   * @param keys vector of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  RmiGIndRobust(const std::vector<key_type> &keys, const std::size_t layer2_size, const float outliers)
      : RmiGIndRobust(keys.begin(), keys.end(), layer2_size, outliers) { }

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
   * @param first, last iterators that define the range of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  template<typename RandomIt>
  RmiGIndRobust(RandomIt first, RandomIt last, const std::size_t layer2_size, const float outliers) : base_type(first, last, layer2_size, outliers) {
    // Compute global absolute errror bounds.
    error_lo_ = 0;
    error_hi_ = 0;
    for (std::size_t i = 0; i != base_type::n_keys_; ++i) {
      key_type key = *(first + i);
      std::size_t segment_id = base_type::get_segment_id(key);
      std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
      if (pred > i) { // overestimation
        error_lo_ = std::max(error_lo_, pred - i);
        base_type::avg_error_ =  i * base_type::avg_error_ / (i + 1) + static_cast<float>(pred - i) / (i + 1);
      } else { // underestimation
        error_hi_ = std::max(error_hi_, i - pred);
        base_type::avg_error_ =  i * base_type::avg_error_ / (i + 1) + static_cast<float>(i - pred) / (i + 1);
      }
    }
  }

  /**
   * Returns a position estimate and search bounds for a given key.
   * @param key to search for
   * @return position estimate and search bounds
   */
  Approx search(const key_type key) const {
    auto segment_id = base_type::get_segment_id(key);
    std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
    std::size_t lo = pred > error_lo_ ? pred - error_lo_ : 0;
    std::size_t hi = std::min(pred + error_hi_ + 1, base_type::n_keys_);
    return {pred, lo, hi};
  }

  /**
   * Returns the size of the index in bytes.
   * @return index size in bytes
   */
  std::size_t size_in_bytes() { return base_type::size_in_bytes() + sizeof(error_lo_) + sizeof(error_hi_); }
};


/**
 * Recursive model index with local absolute bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiLAbsRobust : public RmiRobust<Key, Layer1, Layer2>
{
  using base_type = RmiRobust<Key, Layer1, Layer2>;
  using key_type = Key;
  using layer1_type = Layer1;
  using layer2_type = Layer2;

 protected:
  std::vector<std::size_t> errors_; ///< The error bounds of the layer2 models.

 public:
  /**
   * Default constructor.
   */
  RmiLAbsRobust() = default;

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
   * @param keys vector of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  RmiLAbsRobust(const std::vector<key_type> &keys, const std::size_t layer2_size, const float outliers)
      : RmiLAbsRobust(keys.begin(), keys.end(), layer2_size, outliers) { }

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
   * @param first, last iterators that define the range of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  template<typename RandomIt>
  RmiLAbsRobust(RandomIt first, RandomIt last, const std::size_t layer2_size, const float outliers) : base_type(first, last, layer2_size, outliers) {
    // Compute local absolute error bounds.
    errors_ = std::vector<std::size_t>(layer2_size);
    for (std::size_t i = 0; i != base_type::n_keys_; ++i) {
      key_type key = *(first + i);
      std::size_t segment_id = base_type::get_segment_id(key);
      std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
      if (pred > i) { // overestimation
        errors_[segment_id] = std::max(errors_[segment_id], pred - i);
        base_type::avg_error_ = i * base_type::avg_error_ / (i + 1) + static_cast<float>(pred - i) / (i + 1);
      } else { // underestimation
        errors_[segment_id] = std::max(errors_[segment_id], i - pred);
        base_type::avg_error_ = i * base_type::avg_error_ / (i + 1) + static_cast<float>(i - pred) / (i + 1);
      }
    }
  }

  /**
   * Returns a position estimate and search bounds for a given key.
   * @param key to search for
   * @return position estimate and search bounds
   */
  Approx search(const key_type key) const {
    auto segment_id = base_type::get_segment_id(key);
    std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
    std::size_t err = errors_[segment_id];
    std::size_t lo = pred > err ? pred - err : 0;
    std::size_t hi = std::min(pred + err + 1, base_type::n_keys_);
    return {pred, lo, hi};
  }

  /**
   * Returns the size of the index in bytes.
   * @return index size in bytes
   */
  std::size_t size_in_bytes() { return base_type::size_in_bytes() + errors_.size() * sizeof(errors_.front()); }
};


/**
 * Recursive model index with local individual bounds.
 */
template<typename Key, typename Layer1, typename Layer2>
class RmiLIndRobust : public RmiRobust<Key, Layer1, Layer2>
{
  using base_type = RmiRobust<Key, Layer1, Layer2>;
  using key_type = Key;
  using layer1_type = Layer1;
  using layer2_type = Layer2;

 protected:
  /**
   * Struct to store a lower and an upper error bound.
   */
  struct bounds {
    std::size_t lo; ///< The lower error bound.
    std::size_t hi; ///< The upper error bound.

    /**
     * Default constructor.
     */
    bounds() : lo(0), hi(0) { }
  };

  std::vector<bounds> errors_; ///< The error bounds of the layer2 models.

 public:
  /**
   * Default constructor.
   */
  RmiLIndRobust() = default;

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted @p keys.
   * @param keys vector of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  RmiLIndRobust(const std::vector<key_type> &keys, const std::size_t layer2_size, const float outliers)
      : RmiLIndRobust(keys.begin(), keys.end(), layer2_size, outliers) { }

  /**
   * Builds the index with @p layer2_size models in layer2 on the sorted keys in the range [first, last).
   * @param first, last iterators that define the range of sorted keys to be indexed
   * @param layer2_size the number of models in layer2
   */
  template<typename RandomIt>
  RmiLIndRobust(RandomIt first, RandomIt last, const std::size_t layer2_size, const float outliers) : base_type(first, last, layer2_size, outliers) {
    // Compute local individual errror bounds.
    errors_ = std::vector<bounds>(layer2_size);
    for (std::size_t i = 0; i != base_type::n_keys_; ++i) {
      key_type key = *(first + i);
      std::size_t segment_id = base_type::get_segment_id(key);
      std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
      if (pred > i) { // overestimation
        std::size_t &lo = errors_[segment_id].lo;
        lo = std::max(lo, pred - i);
        base_type::avg_error_ =  i * base_type::avg_error_ / (i + 1) + static_cast<float>(pred - i) / (i + 1);
      } else { // underestimation
        std::size_t &hi = errors_[segment_id].hi;
        hi = std::max(hi, i - pred);
        base_type::avg_error_ =  i * base_type::avg_error_ / (i + 1) + static_cast<float>(i - pred) / (i + 1);
      }
    }
  }

  /**
   * Returns a position estimate and search bounds for a given key.
   * @param key to search for
   * @return position estimate and search bounds
   */
  Approx search(const key_type key) const {
    auto segment_id = base_type::get_segment_id(key);
    std::size_t pred = std::clamp<double>(base_type::l2_[segment_id].predict(key), 0, base_type::n_keys_ - 1);
    bounds err = errors_[segment_id];
    std::size_t lo = pred > err.lo ? pred - err.lo : 0;
    std::size_t hi = std::min(pred + err.hi + 1, base_type::n_keys_);
    return {pred, lo, hi};
  }

  /**
   * Returns the size of the index in bytes.
   * @return index size in bytes
   */
  std::size_t size_in_bytes() { return base_type::size_in_bytes() + errors_.size() * sizeof(errors_.front()); }
};

} // namespace rmi
