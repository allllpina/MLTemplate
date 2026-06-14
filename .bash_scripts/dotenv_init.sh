#!/usr/bin/env bash

set -e

echo "Dotenv initialization"

if [ ! -f .env.example ]; then
    echo "Error: .env.example not found!"
    exit 1
fi

if [ ! -f .env ]; then
    echo "Creating .env file"
    cp .env.example .env
    echo ".env file has been created. Please insert variables"
else
    echo ".env file already exsits"
fi

if [ -f .gitignore ]; then
    if ! grep -q "^\.env$" .gitignore; then
        echo -e ".env" >> .gitignore
        echo ".env has been added to .gitignore"
    fi
else
    echo "# Секрети\n.env" > .gitignore
    echo ".gitignore has been created and modified with .env"
fi


