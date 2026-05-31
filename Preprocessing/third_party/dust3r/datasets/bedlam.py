import os.path as osp
import numpy as np
import os
import sys
import pickle
sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2

invalid_seqs = [
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000042",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000059",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000079",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000978",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000081",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000268",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000089",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000189",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000034",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000889",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000293",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000067",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000904",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000434",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000044",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000013",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000396",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000012",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000082",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000120",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000324",
    "20221013_3_250_batch01hand_static_bigOffice_seq_000038",
    "20221012_3-10_500_batch01hand_zoom_highSchoolGym_seq_000486",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000421",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000226",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000012",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000149",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000311",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000080",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000122",
    "20221012_3-10_500_batch01hand_zoom_highSchoolGym_seq_000079",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000077",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000095",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000062",
    "20221013_3_250_batch01hand_static_bigOffice_seq_000015",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000095",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000119",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000297",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000011",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000196",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000316",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000283",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000085",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000287",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000163",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000804",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000842",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000027",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000182",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000982",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000029",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000031",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000025",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000250",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000785",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000069",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000122",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000246",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000352",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000425",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000192",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000900",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000043",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000063",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000096",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000091",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000013",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000309",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000114",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000969",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000361",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000267",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000083",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000383",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000890",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000003",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000045",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000317",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000076",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000082",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000907",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000279",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000076",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000004",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000061",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000811",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000800",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000841",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000794",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000308",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000064",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000284",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000752",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000269",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000036",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000419",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000290",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000322",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000818",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000327",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000326",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000002",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000060",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000348",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000059",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000016",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000817",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000332",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000094",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000193",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000779",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000177",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000368",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000023",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000024",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000310",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000086",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000038",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000071",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000768",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000017",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000053",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000097",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000856",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000827",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000161",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000084",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000106",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000207",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000007",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000013",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000251",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000796",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000105",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000251",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000046",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000334",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000453",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000373",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000283",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000249",
]
hdri_scenes = [
    "20221010_3_1000_batch01hand",
    "20221017_3_1000_batch01hand",
    "20221018_3-8_250_batch01hand",
    "20221019_3_250_highbmihand",
]


class BEDLAM_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        self.max_humans = 10
        self.smpl_key2shape= {
            'smplx_root_pose': (1, 3), 
            'smplx_body_pose': (21, 3), 
            'smplx_jaw_pose': (1, 3), 
            'smplx_leye_pose': (1, 3), 
            'smplx_reye_pose': (1, 3), 
            'smplx_left_hand_pose': (15, 3), 
            'smplx_right_hand_pose': (15, 3), 
            'smplx_shape': (11,), 
            'smplx_transl': (3,), 
            'smplx_gender_id': (),
            }

        super().__init__(*args, **kwargs)

        if split == "train":
            self.split = "Training"
        elif split == "test":
            self.split = "Test"
        else:
            raise ValueError("")
        
        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        self.scenes = os.listdir(osp.join(self.ROOT, split))

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            if scene in invalid_seqs:
                continue
            if any([scene.startswith(x) for x in hdri_scenes]):
                continue
            if "closeup" in scene:
                continue
            scene_dir = osp.join(self.ROOT, split, scene)
            rgb_dir = osp.join(scene_dir, "rgb")
            basenames = sorted(
                [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")]
            )
            num_imgs = len(basenames)
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )
            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            start_img_ids.extend(start_img_ids_)
            sceneids.extend([j] * num_imgs)
            images.extend(basenames)
            scenes.append(scene)
            scene_img_list.append(img_ids)

            # offset groups
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=1.0,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.split, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            mask_dir = osp.join(scene_dir, "mask")
            cam_dir = osp.join(scene_dir, "cam")
            smpl_dir = osp.join(scene_dir, "smpl")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
            # Load mask image
            if os.path.exists(mask_dir):
                mask_image = imread_cv2(osp.join(mask_dir, basename + ".png"))
            else:
                mask_image = None
            # Load depthmap
            depthmap = np.load(osp.join(depth_dir, basename + ".npy"))
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            depthmap[depthmap > 200.0] = 0.0

            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]

            annot_file = osp.join(smpl_dir, f"{basename}.pkl")
            annots = []
            smpl_mask = np.zeros(self.max_humans, dtype=np.bool_)

            if os.path.isfile(annot_file):
                with open(annot_file, 'rb') as f:
                    annots = pickle.load(f)
                humans = [hum for hum in annots if hum['smplx_transl'][-1] > 0.01] # the person should be in front of the camera
                if len(humans) > 0:
                    smpl_mask[:len(humans)] = 1.
                    l_dist = [hum['smplx_transl'][-1] for hum in humans]
                    indexed_lst = list(enumerate(l_dist))
                    sorted_indexed = sorted(indexed_lst, key=lambda x: x[1], reverse=False)
                    sorted_indices = [index for index, _ in sorted_indexed]
                    annots = [humans[h_idx] for h_idx in sorted_indices]

                    # Update smplx_gender - 0=neutral - 1=male - 2=female - kids?
                    for hum in annots:
                        hum['smplx_gender_id'] = np.asarray({'neutral': 0}[hum['smplx_gender']])

            if mask_image is not None:
                rgb_image, depthmap, mask_image, intrinsics = self._crop_resize_if_necessary_mask(
                    rgb_image, depthmap, mask_image, intrinsics, resolution, rng=rng, info=view_idx
                )
            else:
                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.00, 0.15]
            )
            # Reorganize the smpl annotations
            smpl_dict = {}
            for k in self.smpl_key2shape.keys():
                smpl_dict[k] = np.zeros((self.max_humans, *self.smpl_key2shape[k]), dtype=np.float32)
                if len(humans) > 0:
                    for h in range(len(humans)):
                        smpl_dict[k][h] = annots[h][k].astype(np.float32)

            views.append(
                dict(
                    img=rgb_image,
                    msk=False if mask_image is None else mask_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="BEDLAM",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".png"),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(1, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                    smpl_mask=smpl_mask,
                    **smpl_dict,
                )
            )

        assert len(views) == num_views
        return views
