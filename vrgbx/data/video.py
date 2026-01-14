import imageio, os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Optional, Dict


class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()


def split_file_name(file_name):
    result = []
    number = -1
    for i in file_name:
        if ord(i)>=ord("0") and ord(i)<=ord("9"):
            if number == -1:
                number = 0
            number = number*10 + ord(i) - ord("0")
        else:
            if number != -1:
                result.append(number)
                number = -1
            result.append(i)
    if number != -1:
        result.append(number)
    result = tuple(result)
    return result


def search_for_images(folder):
    file_list = [i for i in os.listdir(folder) if i.endswith(".jpg") or i.endswith(".png")]
    file_list = [(split_file_name(file_name), file_name) for file_name in file_list]
    file_list = [i[1] for i in sorted(file_list)]
    file_list = [os.path.join(folder, i) for i in file_list]
    return file_list


class LowMemoryImageFolder:
    def __init__(self, folder, file_list=None):
        if file_list is None:
            self.file_list = search_for_images(folder)
        else:
            self.file_list = [os.path.join(folder, file_name) for file_name in file_list]
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        return Image.open(self.file_list[item]).convert("RGB")

    def __del__(self):
        pass



def crop_and_resize(image, height, width):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    if image_height / image_width < height / width:
        croped_width = int(image_height / height * width)
        left = (image_width - croped_width) // 2
        image = image[:, left: left+croped_width]
        image = Image.fromarray(image).resize((width, height))
    else:
        croped_height = int(image_width / width * height)
        left = (image_height - croped_height) // 2
        image = image[left: left+croped_height, :]
        image = Image.fromarray(image).resize((width, height))
    return image


class VideoXRGBData:
    DEFAULT_CHANNEL_MAP: Dict[str, str] = {
        "rgb": "rgb.png",
        "albedo": "base_color.png",
        "normal": "normal_cameraspace.png",
        "depth": "depth.png",
        "irradiance": "irradiance.png",
        "material": "material.png",
        "roughness": "material.png",     
        "metallic": "material.png",
        "transparency": "material.png",
    }

    def __init__(
        self,
        root_dir: str = 'xxx',
        channels: str = ['albedo', 'normal', 'material', 'irradiance', 'rgb'],
        start_frame: int = 0,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        channel_map: Optional[Dict[str, str]] = None,
        strict: bool = True,   
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.prefix = os.path.basename(self.root_dir) 
        self.channels = channels
        self.channel_map = dict(self.DEFAULT_CHANNEL_MAP)
        if channel_map:
            self.channel_map.update(channel_map)

        for channel in self.channels:
            if channel not in self.channel_map:
                raise ValueError(f"Unknown channel '{channel}'. "
                                 f"Known: {list(self.channel_map.keys())}")

        self.start_frame = int(start_frame)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.strict = strict

        if self.num_frames is None:
            self.num_frames = self._infer_num_frames(self.channels[0]) 

        self.index_list = list(range(self.start_frame, self.start_frame + self.num_frames))
        self.paths = {}
        for channel in self.channels:
            self.paths[channel] = [self._frame_channel_path(idx, channel) for idx in self.index_list]

        if self.strict:
            for channel in self.channels:
                missing = [p for p in self.paths[channel] if not os.path.isfile(p)]
                if missing:
                    raise FileNotFoundError(
                        f"Missing {len(missing)} frames for channel '{channel}'. "
                        f"E.g. first missing: {missing[0]}"
                    )
        else:
            for channel in self.channels:
                valid = [(idx, p) for idx, p in zip(self.index_list, self.paths[channel]) if os.path.isfile(p)]
                self.index_list = [idx for idx, _ in valid]
                self.paths[channel] = [p for _, p in valid]
                
    def _frame_dir(self, idx: int) -> str:
        return os.path.join(self.root_dir, f"{self.prefix}_frame{idx:04d}")

    def _frame_channel_path(self, idx: int, channel: str) -> str:
        fname = self.channel_map[channel]
        return os.path.join(self._frame_dir(idx), fname)

    def _infer_num_frames(self,channel) -> int:
        count = 0
        idx = self.start_frame
        while True:
            if os.path.isfile(self._frame_channel_path(idx,channel)):
                count += 1
                idx += 1
            else:
                break
        if count == 0:
            raise FileNotFoundError(
                f"No frames found from start_frame={self.start_frame} "
                f"for channel '{channel}' under {self.root_dir}"
            )
        return count

    def __len__(self):
        return len(self.paths[self.channels[0]])

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        img = self.__getitem__(0)
        h, w = img.size[1], img.size[0]  # PIL size=(w,h)
        return h, w

    def __getitem__(self, i: int) -> Image.Image:
        imgs = []
        for channel in self.channels:
            path = self.paths[channel][i]
            img = Image.open(path).convert("RGB")
            if self.height is not None and self.width is not None:
                w, h = img.size
                if (h != self.height) or (w != self.width):
                    img = crop_and_resize(img, self.height, self.width)
            imgs.append(img)
        return imgs

    def raw_data(self) -> List[Image.Image]:
        return [self[i] for i in range(len(self))]

    def save_images(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(len(self)), desc=f"Saving {self.channel}"):
            frame = self[i]
            frame.save(os.path.join(folder, f"{self.index_list[i]:04d}.png"))


class VideoRGBXData:

    DEFAULT_CHANNEL_MAP: Dict[str, str] = {
        "rgb": "rgb.png",
        "base_color": "base_color.png",
        "normal": "normal.png",
        "normal_cameraspace": "normal_cameraspace.png",
        "depth": "depth.png",
        "irradiance": "irradiance.png",
        "material": "material.png",
        "roughness": "material.png",      
        "metallic": "material.png",
        "transparency": "material.png",
    }

    def __init__(
        self,
        root_dir: str = '',
        channel: str = "rgb",
        start_frame: int = 0,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        channel_map: Optional[Dict[str, str]] = None,
        strict: bool = True,   
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.prefix = os.path.basename(self.root_dir)  
        self.channel = channel
        self.channel_map = dict(self.DEFAULT_CHANNEL_MAP)
        if channel_map:
            self.channel_map.update(channel_map)

        if self.channel not in self.channel_map:
            raise ValueError(f"Unknown channel '{self.channel}'. "
                             f"Known: {list(self.channel_map.keys())}")

        self.start_frame = int(start_frame)
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.strict = strict

        if self.num_frames is None:
            self.num_frames = self._infer_num_frames()

        self.index_list = list(range(self.start_frame, self.start_frame + self.num_frames))
        self.paths = [self._frame_channel_path(idx) for idx in self.index_list]

        if self.strict:
            missing = [p for p in self.paths if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(
                    f"Missing {len(missing)} frames for channel '{self.channel}'. "
                    f"E.g. first missing: {missing[0]}"
                )
        else:
            kept = []
            kept_idx = []
            for idx, p in zip(self.index_list, self.paths):
                if os.path.isfile(p):
                    kept.append(p)
                    kept_idx.append(idx)
            self.paths = kept
            self.index_list = kept_idx

    def _frame_dir(self, idx: int) -> str:
        return os.path.join(self.root_dir, f"{self.prefix}_frame{idx:04d}")

    def _frame_channel_path(self, idx: int) -> str:
        fname = self.channel_map[self.channel]
        return os.path.join(self._frame_dir(idx), fname)

    def _infer_num_frames(self) -> int:
        count = 0
        idx = self.start_frame
        while True:
            if os.path.isfile(self._frame_channel_path(idx)):
                count += 1
                idx += 1
            else:
                break
        if count == 0:
            raise FileNotFoundError(
                f"No frames found from start_frame={self.start_frame} "
                f"for channel '{self.channel}' under {self.root_dir}"
            )
        return count

    def __len__(self):
        return len(self.paths)

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        img = self.__getitem__(0)
        h, w = img.size[1], img.size[0]  # PIL size=(w,h)
        return h, w

    def __getitem__(self, i: int) -> Image.Image:
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        if self.height is not None and self.width is not None:
            w, h = img.size
            if (h != self.height) or (w != self.width):
                img = crop_and_resize(img, self.height, self.width)
        return img

    def raw_data(self) -> List[Image.Image]:
        return [self[i] for i in range(len(self))]

    def save_images(self, folder: str):
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(len(self)), desc=f"Saving {self.channel}"):
            frame = self[i]
            frame.save(os.path.join(folder, f"{self.index_list[i]:04d}.png"))

class DummyVideo:
    def __init__(self, num_frames, height, width, value=0):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.value = value

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        arr = np.full((self.height, self.width, 3), self.value, dtype=np.uint8)
        return Image.fromarray(arr)

class VideoData:
    def __init__(self, video_file=None, image_folder=None, height=None, width=None, dummy=False, num_frames=49, fill_value=0, **kwargs):
        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        elif dummy:
            self.data_type = "dummy"
            if height is None or width is None:
                raise ValueError("Dummy video requires height and width")
            self.data = DummyVideo(num_frames, height, width, value=fill_value)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.set_shape(height, width)

    def raw_data(self):
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length):
        self.length = length

    def set_shape(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        else:
            height, width, _ = self.__getitem__(0).shape
            return height, width

    def __getitem__(self, item):
        frame = self.data.__getitem__(item)
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = crop_and_resize(frame, self.height, self.width)
        return frame

    def __del__(self):
        pass

    def save_images(self, folder):
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(self.__len__()), desc="Saving images"):
            frame = self.__getitem__(i)
            frame.save(os.path.join(folder, f"{i}.png"))


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

def save_frames(frames, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, frame in enumerate(tqdm(frames, desc="Saving images")):
        frame.save(os.path.join(save_path, f"{i}.png"))
