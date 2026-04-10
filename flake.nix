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
          buildInputs = commonBuildInputs ++ (with pkgs; [
            cudaPackages.cudatoolkit
            vulkan-loader  # For wgpu/TestRuntime in cubek tests
          ]);
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs
            + ":/run/opengl-driver/lib"  # System NVIDIA driver (libcuda.so)
            + ":$LD_LIBRARY_PATH";
          NIX_LD = "${pkgs.stdenv.cc.libc}/lib/ld-linux-x86-64.so.2";
          CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
        };
      }
    );
}
