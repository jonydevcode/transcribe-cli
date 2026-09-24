{
  description = "ROCm development environment for transcribe-cli";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python313.withPackages (ps: [ ps.huggingface-hub ]);
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          python
          ffmpeg
          cmake
          git
          vulkan-headers
          vulkan-loader
          vulkan-tools
          shaderc
        ];

        shellHook = ''
          echo "transcribe-cli Vulkan GPU shell"
          echo "Run: python main.py INPUT.EXT"
        '';
      };
    };
}
