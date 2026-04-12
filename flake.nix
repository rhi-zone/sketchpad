{
  description = "sketchpad";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Common build inputs for all shells
        commonBuildInputs = with pkgs; [
          stdenv.cc.cc
          # Rust toolchain
          rustc
          cargo
          rust-analyzer
          clippy
          rustfmt
          # Fast linker for incremental builds
          mold
          clang
          # JS tooling for docs
          bun
        ];

        # Vulkan loader + system ICD path — needed for wgpu on any GPU
        # /run/opengl-driver is where NixOS places system GPU userspace drivers
        # (libdrm_amdgpu.so, radeon_icd.x86_64.json, nvidia_icd.json, etc.)
        vulkanInputs = with pkgs; [ vulkan-loader vulkan-headers ];
        vulkanLibPath = "/run/opengl-driver/lib";
      in
      {
        # Default shell without GPU dependencies
        devShells.default = pkgs.mkShell rec {
          buildInputs = commonBuildInputs;
          LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath buildInputs}:$LD_LIBRARY_PATH";
          # tracel-llvm ships pre-built ELF binaries that need a real dynamic linker
          NIX_LD = "${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2";
        };

        # CUDA shell - use with `nix develop .#cuda`
        # Requires system NVIDIA drivers (libcuda.so from /run/opengl-driver)
        devShells.cuda = pkgs.mkShell rec {
          buildInputs = commonBuildInputs ++ vulkanInputs ++ (with pkgs; [
            cudaPackages.cudatoolkit
          ]);
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs
            + ":${vulkanLibPath}"  # System NVIDIA driver (libcuda.so) + Vulkan ICD
            + ":$LD_LIBRARY_PATH";
          NIX_LD = "${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2";
          CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
        };

        # AMD shell - use with `nix develop .#amd`
        # Covers both wgpu/Vulkan (default) and ROCm/HIP (--features rocm).
        # ROCm note: libamd_comgr JIT-compiles GPU kernels and can crash on init
        # if the system ROCm stack is incomplete; wgpu is a reliable fallback.
        devShells.amd = pkgs.mkShell rec {
          buildInputs = commonBuildInputs ++ vulkanInputs ++ (with pkgs.rocmPackages; [
            clr          # libamdhip64.so + libhiprtc.so (HIP runtime + RTC)
            hipcc        # hipconfig binary (required by cubecl-hip-sys build.rs)
            rocm-runtime # libhsa-runtime64.so (HSA runtime, needed at runtime)
          ]);
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs
            + ":${vulkanLibPath}"  # System AMDGPU driver (libdrm_amdgpu.so, Vulkan ICD)
            + ":$LD_LIBRARY_PATH";
          NIX_LD = "${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2";
          # cubecl-hip-sys build.rs checks HIP_PATH to find libraries
          HIP_PATH = pkgs.rocmPackages.clr;
        };

        # Keep .#rocm as an alias for backwards compatibility
        devShells.rocm = self.outputs.devShells.${system}.amd;
      }
    );
}
