{
  description = "Development environment for transcribe-cli";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          isRocmSystem = system == "x86_64-linux";
          rocmTools = with pkgs.rocmPackages; [
            llvm.clang
            llvm.lld
            rocm-device-libs
            rocminfo
            rocm-smi
          ];
          rocmRoot =
            if isRocmSystem then
              pkgs.symlinkJoin {
                name = "rocm-root";
                paths = with pkgs.rocmPackages; [
                  clr
                  llvm.clang
                  llvm.lld
                  rocm-device-libs
                ];
              }
            else
              null;
        in
        {
          default = pkgs.mkShell {
            packages =
              (with pkgs; [
                python312
                uv
                ffmpeg
                libsndfile
                patchelf
              ])
              ++ pkgs.lib.optionals isRocmSystem rocmTools;

            env = {
              UV_PYTHON_DOWNLOADS = "never";
              UV_PYTHON_PREFERENCE = "only-system";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.bzip2
                pkgs.xz
                pkgs.zlib
                pkgs.zstd
                pkgs.libsndfile
              ];
              HIP_DEVICE_LIB_PATH = pkgs.lib.optionalString isRocmSystem
                "${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode";
              HIP_PATH = pkgs.lib.optionalString isRocmSystem "${rocmRoot}";
              ROCM_PATH = pkgs.lib.optionalString isRocmSystem "${rocmRoot}";
              # Radeon 860M (gfx1152) currently needs the compatible gfx1150
              # code path to avoid a ROCm runtime compiler crash.
              HSA_OVERRIDE_GFX_VERSION =
                pkgs.lib.optionalString isRocmSystem "11.5.0";
            };

            shellHook = ''
              echo "transcribe-cli development shell (${
                if isRocmSystem then "ROCm" else "CPU"
              })"
              echo "Run: uv sync --frozen"
              ${
                if isRocmSystem then
                  ''echo "Then: ./scripts/link-rocm-libraries"''
                else
                  ""
              }
            '';
          };
        });
    };
}
