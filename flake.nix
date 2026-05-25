{
  description = "MLOps Mini-Framework for Neural Network Experiments";

  inputs = {
    # Використовуємо unstable для найсвіжіших версій PyTorch та CUDA-пакетів
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      
      # Налаштовуємо nixpkgs з підтримкою невільного ПЗ (CUDA) та самого CUDA
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      # Збираємо наше Python-середовище з усіма необхідними лібами
      pythonEnv = pkgs.python311.withPackages (ps: with ps; [
        # Deep Learning & Math
        torch
        torchvision
        torchaudio
        pytorch-lightning
        numpy
        pandas
        scikit-learn

        # Experiment Tracking & MLOps
        mlflow
        dvc
        
        # Configuration & CLI
        hydra-core
        typer
        rich # Для красивих логів у терміналі
      ]);

    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "mlops-env";
        
        buildInputs = [
          pythonEnv
          pkgs.git
          pkgs.dvc # DVC як системний пакет для CLI
        ];

        # Налаштовуємо змінні середовища для коректної роботи CUDA в Nix
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib:$LD_LIBRARY_PATH
          export EXTRA_CCFLAGS="-I/usr/include"
          
          echo "========================================="
          echo "MLFramework Environment Activated!"
          echo "Python: $(python --version)"
          echo "GPU Access: $(python -c 'import torch; print("OK" if torch.cuda.is_available() else "No CUDA")')"
          echo "========================================="
        '';
      };
    };
}