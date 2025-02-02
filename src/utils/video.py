import cv2
import imageio
import numpy as np
import torch
from torchvision.transforms import Resize, CenterCrop, Compose
from torchvision.transforms.v2 import Normalize, ToImage, ToDtype
import torch.nn as nn

class VideoVAE(nn.Module):
    def __init__(self):
        super().__init__()
        # from diffusers import AutoencoderKLCogVideoX
        # self.vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float)
        from diffusers import AutoencoderKLLTXVideo
        self.vae = AutoencoderKLLTXVideo.from_pretrained("Lightricks/LTX-Video", subfolder="vae",
                                                    torch_dtype=torch.float32)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    def vae_encode(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
            x = self.vae.encode(x).latent_dist.sample()[0]
        else:
            x = self.vae.encode(x).latent_dist.sample()
        return x

    def vae_decode(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)
            x = self.vae.decode(x).sample[0]
        else:
            x = self.vae.decode(x).sample
        x = (x.clamp(-1, 1) / 2 + 0.5)
        return x.float()


def interpolate_frames(frame1, frame2, alpha):
    # Linear interpolation between two frames
    interpolated_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
    return interpolated_frame


def read_video(video_path, size=(208, 320), target_fps=24, start_frame=0, end_frame=-1):
    # set transforms
    transforms = Compose([
        ToImage(),
        Resize(size=int(np.min(size)*1.2)),
        CenterCrop(size=size),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.5], std=[0.5]),
    ])

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
        return

    source_fps = np.ceil(cap.get(cv2.CAP_PROP_FPS))
    # print(f"source fps: {source_fps}")
    frame_count = 0
    encodings = []

    # Calculate the frame interval for downsampling
    if target_fps < source_fps:
        frame_interval = source_fps / target_fps
    else:
        frame_interval = 1

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading the first frame")
        return


    while True:
        ret, next_frame = cap.read()

        if not ret:
            break

        if frame_count < start_frame:
            frame_count += 1
            continue

        # # Downsample frames if target_fps is smaller than source_fps
        # if target_fps < source_fps:
        #     if frame_count % frame_interval == 0:
        #         # resize
        #         encoded_frame = transforms(next_frame).flip(dims=(0,))
        #
        #         # Store or process the encoded frame as needed
        #         encodings.append(encoded_frame)
        # else:
        #     print("upsample framerate")
        #     # Upsample frames if target_fps is greater than source_fps
        #     num_interpolated_frames = int(source_fps / target_fps)
        #
        #     for i in range(num_interpolated_frames):
        #         alpha = (i + 1) / (num_interpolated_frames + 1)
        #         interpolated_frame = interpolate_frames(prev_frame, next_frame, alpha)
        #
        #         # resize
        #         encoded_frame = transforms(interpolated_frame).flip(dims=(0,))
        #
        #         # Store or process the encoded frame as needed
        #         encodings.append(encoded_frame)
        # resize
        encoded_frame = transforms(next_frame).flip(dims=(0,))
        # Store or process the encoded frame as needed
        encodings.append(encoded_frame)

        # Update the previous frame
        prev_frame = next_frame
        frame_count += 1

        if end_frame>0 and frame_count>=end_frame:
            break

    # Release the video capture object
    cap.release()

    # Return the list of encodings or save them to a file
    video = torch.stack(encodings)
    return video.permute(1, 0, 2, 3), source_fps

def write_video(video, path, target_fps=16):
    video = (video.permute(1,2,3,0)).clamp(0, 1) * 255
    video = video.type(torch.uint8).numpy()

    with imageio.get_writer(path, fps=target_fps) as writer:
        for frame in video:
            writer.append_data(frame)


def vae_encode_video(video, vae, temp_chunk_size=240):
    encoded = []
    l = video.shape[1]
    # q = l // temp_chunk_size
    # for i in range(q):
    #     encoded.append(vae.vae_encode(video[:, i * temp_chunk_size:(i + 1) * temp_chunk_size, ...]))
    # if q*temp_chunk_size < l:
    #     encoded.append(vae.vae_encode(video[:, q * temp_chunk_size:l, ...]))
    encoded.append(vae.vae_encode(video))
    return torch.cat(encoded, dim=1)

def vae_decode_video(video, vae, batch_size=2):
    # decoded = []
    # l = video.shape[1]
    # q = l // batch_size
    # for i in range(q):
    #     decoded.append(vae.vae_decode(video[:, i*batch_size:(i+1)*batch_size, ...]))
    # if q * batch_size < l:
    #     decoded.append(vae.vae_decode(video[:, q*batch_size:l, ...]))
    # return torch.cat(decoded, dim=1)
    return vae.vae_decode(video)


if __name__ == "__main__":
    import sys
    sys.path.append("src/")

    fps = 24

    vae = VideoVAE().to("cuda")
    video, fps = read_video(sys.argv[1], start_frame=0, end_frame=121, size=(160,256), target_fps=fps)
    video = video.to("cuda")
    print(f"video size: {video.shape}")
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        encoded = vae_encode_video(video, vae)
        print(f"encoded: {encoded.shape}")
        encoded = encoded
        decoded = vae_decode_video(encoded, vae).detach().cpu()
        print(f"decoded: {decoded.shape}")

    import matplotlib.pyplot as plt
    plt.ion()
    decoded = (decoded.permute(1,0,2,3))
    for f in decoded:
        plt.clf()
        plt.imshow(f.permute(1, 2, 0))
        plt.show()
        plt.pause(0.04)


    write_video(decoded.permute(1,0,2,3), sys.argv[2], target_fps=fps)