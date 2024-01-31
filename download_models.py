from susie.model import load_vae, load_text_encoder


pretrained_path = "runwayml/stable-diffusion-v1-5:flax"
vae_encode, vae_decode = load_vae(pretrained_path)
tokenize, untokenize, text_encode = load_text_encoder(pretrained_path)

print("Done downloading models.")