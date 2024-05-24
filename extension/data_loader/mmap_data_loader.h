/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/types.h>
#endif

namespace torch {
namespace executor {
namespace util {

/**
 * A DataLoader that loads segments from a file, allocating the memory
 * with `malloc()`.
 *
 * Note that this will keep the file open for the duration of its lifetime, to
 * avoid the overhead of opening it again for every Load() call.
 */
class MmapDataLoader : public DataLoader {
 public:
  /**
   * Describes how and whether to lock loaded pages with `mlock()`.
   *
   * Using `mlock()` typically loads all of the pages immediately, and will
   * typically ensure that they are not swapped out. The actual behavior
   * will depend on the host system.
   */
  enum class MlockConfig {
    /// Do not call `mlock()` on loaded pages.
    NoMlock,
    /// Call `mlock()` on loaded pages, failing if it fails.
    UseMlock,
    /// Call `mlock()` on loaded pages, ignoring errors if it fails.
    UseMlockIgnoreErrors,
  };

  /**
   * Creates a new MmapDataLoader that wraps the named file. Fails if
   * the file can't be opened for reading or if its size can't be found.
   *
   * @param[in] file_name The path to the file to load from. The file will be
   *     kept open until the MmapDataLoader is destroyed, to avoid the
   *     overhead of opening it again for every Load() call.
   * @param[in] mlock_config How and whether to lock loaded pages with
   *     `mlock()`.
   */
  static Result<MmapDataLoader> from(
      const char* file_name,
      MlockConfig mlock_config = MlockConfig::UseMlock);

  /// DEPRECATED: Use the lowercase `from()` instead.
  __ET_DEPRECATED static Result<MmapDataLoader> From(
      const char* file_name,
      MlockConfig mlock_config = MlockConfig::UseMlock) {
    return from(file_name, mlock_config);
  }

  /// DEPRECATED: Use the version of `from()` that takes an MlockConfig.
  __ET_DEPRECATED
  static Result<MmapDataLoader> From(const char* file_name, bool use_mlock) {
    MlockConfig mlock_config =
        use_mlock ? MlockConfig::UseMlock : MlockConfig::NoMlock;
    return from(file_name, mlock_config);
  }

  // Movable to be compatible with Result.
  MmapDataLoader(MmapDataLoader&& rhs) noexcept
      : file_name_(rhs.file_name_),
        file_size_(rhs.file_size_),
        page_size_(rhs.page_size_),
#ifdef _WIN32
        file_handle_(rhs.file_handle_),
        mapping_handle_(rhs.mapping_handle_),
#else
        fd_(rhs.fd_),
#endif
        mlock_config_(rhs.mlock_config_) {
    rhs.file_name_ = nullptr;
    rhs.file_size_ = 0;
    rhs.page_size_ = 0;
#ifdef _WIN32
    rhs.file_handle_ = INVALID_HANDLE_VALUE;
    rhs.mapping_handle_ = nullptr;
#else
    rhs.fd_ = -1;
#endif
    rhs.mlock_config_ = MlockConfig::NoMlock;
  }

  ~MmapDataLoader() override;

  __ET_NODISCARD Result<FreeableBuffer> Load(size_t offset, size_t size)
      override;

  __ET_NODISCARD Result<size_t> size() const override;

 private:
  MmapDataLoader(
#ifdef _WIN32
      HANDLE file_handle,
      HANDLE mapping_handle,
#else
      int fd,
#endif
      size_t file_size,
      const char* file_name,
      size_t page_size,
      MlockConfig mlock_config)
      : file_name_(file_name),
        file_size_(file_size),
        page_size_(page_size),
#ifdef _WIN32
        file_handle_(file_handle),
        mapping_handle_(mapping_handle),
#else
        fd_(fd),
#endif
        mlock_config_(mlock_config) {}

  // Not safely copyable.
  MmapDataLoader(const MmapDataLoader&) = delete;
  MmapDataLoader& operator=(const MmapDataLoader&) = delete;
  MmapDataLoader& operator=(MmapDataLoader&&) = delete;

  const char* file_name_; // String data is owned by the instance.
  size_t file_size_;
  size_t page_size_;
#ifdef _WIN32
  HANDLE file_handle_;
  HANDLE mapping_handle_;
#else
  int fd_; // Owned by the instance.
#endif
  MlockConfig mlock_config_;
};

} // namespace util
} // namespace executor
} // namespace torch
