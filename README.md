# VGG Deploy 
This project trains a simple VGG-style CNN on CIFAR-10 and deploys it as an API.

## Project Goals
- Implement TinyVGG in PyTorch
- Train and save model artifacts
- Expose inference via FastAPI
- Containerize with Docker
- Deploy to the cloud

## Project Structure
vgg-deploy/
src/ # model and training code
api/ # FastAPI app
scripts/ # helper scripts
configs/ # yaml/json config files
tests/ # unit tests
artifacts/ # saved models and mappings


## Next Steps
- [ ] Implement TinyVGG
- [ ] Write training loop
- [ ] Build inference script
- [ ] Add FastAPI API
- [ ] Dockerize and deploy
