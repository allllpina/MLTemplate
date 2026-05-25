{
  description = "MLFramework Environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { 
        inherit system;
        config.allowUnfree = true;
        };
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "ml-framework";
        
        buildInputs = [
          pkgs.python312
          pkgs.python312Packages.pip
          pkgs.python312Packages.virtualenv
          pkgs.git
          pkgs.dvc
          pkgs.zlib
          pkgs.stdenv.cc.cc.lib
          pkgs.linuxPackages.nvidia_x11
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
          
          if [ ! -d ".venv" ]; then
            echo "Creating virtual environment..."
            python -m venv .venv
          fi
          
          source .venv/bin/activate
          
          echo "========================================="
          echo "Run the following to install packages:"
          echo "pip install torch torchvision pytorch-lightning numpy pandas scikit-learn mlflow dvc hydra-core typer rich"
          echo "========================================="
        '';
      };
    };
}