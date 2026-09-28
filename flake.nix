{
  description = "GPU-only transcribe.cpp frontend (Vulkan)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python313;
      pyprojectToml = builtins.fromTOML (builtins.readFile ./pyproject.toml);

      # The pinned transcribe.cpp commit lives here and nowhere else. Re-test
      # Parakeet/Nemotron batching (`supports_batching` in models.py) when bumping it.
      transcribe-cpp = pkgs.stdenv.mkDerivation {
        pname = "transcribe-cpp";
        version = "0.2.3";
        src = pkgs.fetchFromGitHub {
          owner = "handy-computer";
          repo = "transcribe.cpp";
          rev = "c83df3f229accd6f769cb91b638cd365f3624c9c";
          hash = "sha256-Ihv0Khns9GbHglgRu8bJmMZcm4Ft4/XCE1kivFWp8x8=";
        };
        nativeBuildInputs = with pkgs; [ cmake shaderc pkg-config ];
        buildInputs = with pkgs; [ vulkan-headers vulkan-loader ];
        cmakeFlags = [
          "-DTRANSCRIBE_VULKAN=ON"
          "-DTRANSCRIBE_BUILD_TESTS=OFF"
          "-DCMAKE_BUILD_TYPE=Release"
        ];
        buildPhase = ''
          runHook preBuild
          cmake --build . --target transcribe-cli -j $NIX_BUILD_CORES
          runHook postBuild
        '';
        installPhase = ''
          runHook preInstall
          install -Dm755 bin/transcribe-cli $out/bin/transcribe-cli
          runHook postInstall
        '';
      };

      transcribe-cli = python.pkgs.buildPythonApplication {
        pname = "transcribe-cli";
        inherit (pyprojectToml.project) version;
        pyproject = true;
        src = ./.;
        build-system = [ python.pkgs.setuptools ];
        dependencies = [ python.pkgs.huggingface-hub ];
        makeWrapperArgs = [
          "--set-default TRANSCRIBE_CPP_BIN ${transcribe-cpp}/bin/transcribe-cli"
          "--prefix PATH : ${pkgs.lib.makeBinPath [ pkgs.ffmpeg ]}"
        ];
        doCheck = false; # tests run in checks.default
      };

      devPython = python.withPackages (ps: [
        ps.huggingface-hub
        ps.pytest
        ps.mypy
        ps.ruff
      ]);
    in
    {
      packages.${system} = {
        inherit transcribe-cpp transcribe-cli;
        default = transcribe-cli;
      };

      apps.${system}.default = {
        type = "app";
        program = "${transcribe-cli}/bin/transcribe";
      };

      checks.${system}.default = pkgs.runCommand "transcribe-cli-checks" {
        nativeBuildInputs = [ devPython pkgs.ffmpeg ];
        src = ./.;
      } ''
        cp -r $src source
        chmod -R u+w source
        cd source
        export HOME=$TMPDIR
        ruff check .
        mypy
        pytest -q -p no:cacheprovider
        touch $out
      '';

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          devPython
          pkgs.ffmpeg
          pkgs.vulkan-tools
        ];

        TRANSCRIBE_CPP_BIN = "${transcribe-cpp}/bin/transcribe-cli";
        PYTHONPATH = "src";

        shellHook = ''
          echo "transcribe-cli Vulkan GPU shell"
          echo "Run: python -m transcribe_cli INPUT.EXT"
        '';
      };
    };
}
