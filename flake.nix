{
  description = "ROCm development environment for transcribe-cli";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python313.withPackages (pythonPackages: with pythonPackages; [
        torchWithRocm
        transformers
        huggingface-hub
        librosa
        protobuf
        sentencepiece
        soundfile
      ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python
          ffmpeg
          libsndfile
          rocmPackages.rocminfo
          rocmPackages.rocm-smi
        ];

        shellHook = ''
          if rocminfo 2>/dev/null | grep -q 'gfx1152'; then
            # Radeon 860M compatibility: this Torch build contains gfx1150
            # device code, but not gfx1152 device code.
            export HSA_OVERRIDE_GFX_VERSION=11.5.0
            export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
            echo "transcribe-cli ROCm shell (gfx1152 using gfx1150 compatibility)"
          else
            unset HSA_OVERRIDE_GFX_VERSION
            unset TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
            echo "transcribe-cli ROCm shell"
          fi
          echo "Run: python main.py INPUT.EXT"
        '';
      };
    };
}
