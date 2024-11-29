#!/bin/bash

# cd to the directory of the script
cd "$(dirname "$0")"

# Build the project using docker
docker compose build --push

sed -i "s/DEPLOYID-.*$/DEPLOYID-$(openssl rand -hex 16)/" deployment/service.yaml
new_secret=$(openssl rand -hex 16)
kubectl -n kubishi-sentences delete secret secret-key
kubectl -n kubishi-sentences create secret generic secret-key --from-literal=SECRET_KEY="SECRET-$new_secret"

# Apply the deployment
kubectl apply -f deployment/service.yaml
