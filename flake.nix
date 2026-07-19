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
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python312
              uv
              ffmpeg
              libsndfile
            ];

            env = {
              UV_PYTHON_DOWNLOADS = "never";
              UV_PYTHON_PREFERENCE = "only-system";
              LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.libsndfile
              ];
            };

            shellHook = ''
              echo "transcribe-cli development shell (CPU)"
              echo "Run: uv sync --frozen"
            '';
          };
        });
    };
}
