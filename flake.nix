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
        
        buildInputs = with pkgs; [
          tree
          uv
          python312
          git
          dvc
          zlib
          glib
          stdenv.cc.cc.lib
          linuxPackages.nvidia_x11
        ];
        shellHook = ''
          # Додаємо шлях до бінарників віртуального середовища на початок PATH
          export PATH=$PWD/.venv/bin:$PATH
          
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
            pkgs.zlib
            pkgs.glib
            pkgs.stdenv.cc.cc.lib
            pkgs.linuxPackages.nvidia_x11
          ]}:$LD_LIBRARY_PATH
          
          unset PYTHONPATH
          
          if [ -f "pyproject.toml" ]; then
            echo "Syncing dependencies with uv..."
            uv sync
            # Після синхронізації PATH вже оновлено, тому source не є обов'язковим,
            # але активація додасть VIRTUAL_ENV змінні для інструментів
            source .venv/bin/activate
          else
            echo "pyproject.toml не знайдено."
          fi
        '';
         };
    };
}
