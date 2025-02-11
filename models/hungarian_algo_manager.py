
import torch
from scipy.optimize import linear_sum_assignment
from utils import log_info

class HungarianAlgoManager:
    def __init__(self, data_loader, batch_size, match_size, device):
        log_info(f"HungarianAlgoManager::__init__()...")
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.match_size = match_size
        self.device = device
        self.data_iter = iter(self.data_loader)
        self.remain_batch = None            # the remainder of the data iteration
        self.image_set = torch.Tensor([]).to(self.device)   # image set & noise set are paired by Hungarian algorithm
        self.noise_set = torch.Tensor([]).to(self.device)
        log_info(f"  data_loader: {len(self.data_loader)}")
        log_info(f"  batch_size : {self.batch_size}")
        log_info(f"  match_size : {self.match_size}")
        log_info(f"  device     : {self.device}")
        log_info(f"HungarianAlgoManager::__init__()...")

    def __len__(self):
        return len(self.data_loader)

    def reset(self):
        self.data_iter = iter(self.data_loader)
        self.remain_batch = None            # the remainder of the data iteration
        self.image_set = torch.Tensor([]).to(self.device)
        self.noise_set = torch.Tensor([]).to(self.device)

    def get_data_by_match_size(self):
        if self.remain_batch is None:
            image_batch_arr = []
            image_cnt = 0
        else:
            image_batch_arr = [self.remain_batch]
            image_cnt = len(self.remain_batch)
            self.remain_batch = None
        image_batch = next(self.data_iter, None)
        while image_batch is not None:
            image_batch_arr.append(image_batch)
            image_cnt += len(image_batch)
            if image_cnt >= self.match_size:
                break
            image_batch = next(self.data_iter, None)
        # while
        if image_cnt == 0:
            return None
        whole_image_batch = torch.cat(image_batch_arr, dim=0)
        if image_cnt == self.match_size:
            return whole_image_batch
        ret_batch = whole_image_batch[:self.match_size]
        self.remain_batch = whole_image_batch[self.match_size:]
        return ret_batch

    def get_sample_and_noise_batch(self):
        bs = self.batch_size
        remain_image_set = torch.Tensor([]).to(self.device)
        remain_noise_set = torch.Tensor([]).to(self.device)
        new_image_set = self.get_data_by_match_size()
        while new_image_set is not None:
            new_image_set = new_image_set.to(self.device)
            new_noise_set = self.gen_noise_set(new_image_set)
            self.hungarian_match(new_image_set, new_noise_set)
            new_image_set = torch.cat([remain_image_set, new_image_set], dim=0)
            new_noise_set = torch.cat([remain_noise_set, new_noise_set], dim=0)
            while len(new_image_set) >= bs:
                ret_image = new_image_set[:bs]
                ret_noise = new_noise_set[:bs]
                new_image_set = new_image_set[bs:]
                new_noise_set = new_noise_set[bs:]
                yield ret_image, ret_noise
            remain_image_set = new_image_set
            remain_noise_set = new_noise_set
            new_image_set = self.get_data_by_match_size()
        # while
        if len(remain_image_set) > 0:
            yield remain_image_set, remain_noise_set

    def gen_noise_set(self, image_set):
        # put it in separate function, so we can make unit test easily.
        noise_set = torch.randn_like(image_set, device=self.device)
        return noise_set

    @staticmethod
    def hungarian_match(image_set, noise_set):
        cost_arr = []
        for i in range(len(image_set)):
            x = image_set[i:i + 1]
            distance = (x - noise_set).square().mean(dim=(1, 2, 3))
            cost_arr.append(distance)
        cost_matrix = torch.stack(cost_arr).cpu().numpy()
        _, col_idx = linear_sum_assignment(cost_matrix)
        new_noises = noise_set[col_idx] # it will create new Tensor
        noise_set[:] = new_noises[:]
        del new_noises

# class

if __name__ == '__main__':
    import unittest
    predefined_noise = None
    predefined_noise_set_fn = lambda img_set: predefined_noise
    class HungarianAlgoManagerDerived(HungarianAlgoManager):
        def gen_noise_set(self, image_set):
            return predefined_noise_set_fn(image_set)
    # class HungarianAlgoManagerDerived

    class TestHungarianAlgoManager(unittest.TestCase):
        # @unittest.skip
        def test_get_data_by_match_size1(self):
            arr = [[i] for i in range(1, 6)]
            data_loader = torch.tensor(arr, dtype=torch.int)
            batch_size = 3
            match_size = 2
            device = 'cpu'
            ha = HungarianAlgoManager(data_loader, batch_size, match_size, device)
            set1 = ha.get_data_by_match_size()  # [1, 2]
            set2 = ha.get_data_by_match_size()  # [3, 4]
            set3 = ha.get_data_by_match_size()  # [5]
            set4 = ha.get_data_by_match_size()  # None
            self.assertEqual(len(set1), 2)
            self.assertEqual(set1[0], 1)
            self.assertEqual(set1[1], 2)
            self.assertEqual(len(set2), 2)
            self.assertEqual(set2[0], 3)
            self.assertEqual(set2[1], 4)
            self.assertEqual(len(set3), 1)
            self.assertEqual(set3[0], 5)
            self.assertEqual(set4, None)

        # @unittest.skip
        def test_get_data_by_match_size2(self):
            arr = [[i] for i in range(1, 6)]
            data_loader = torch.tensor(arr, dtype=torch.int)
            batch_size = 2
            match_size = 3
            device = 'cpu'
            ha = HungarianAlgoManager(data_loader, batch_size, match_size, device)
            set1 = ha.get_data_by_match_size()  # [1, 2, 3]
            set2 = ha.get_data_by_match_size()  # [4, 5]
            set3 = ha.get_data_by_match_size()  # None
            set4 = ha.get_data_by_match_size()  # None
            self.assertEqual(len(set1), 3)
            self.assertEqual(set1[0], 1)
            self.assertEqual(set1[1], 2)
            self.assertEqual(set1[2], 3)
            self.assertEqual(len(set2), 2)
            self.assertEqual(set2[0], 4)
            self.assertEqual(set2[1], 5)
            self.assertEqual(set3, None)
            self.assertEqual(set4, None)

        # @unittest.skip
        def test_get_sample_and_noise_batch1(self):
            arr = [[[[[i]]]] for i in range(1, 7)]
            data_loader = torch.tensor(arr, dtype=torch.float)
            batch_size = 6
            match_size = 6
            device = 'cpu'
            ha = HungarianAlgoManager(data_loader, batch_size, match_size, device)
            for images, noises in ha.get_sample_and_noise_batch():
                self.assertEqual(len(images), batch_size)
                self.assertEqual(len(noises), batch_size)

        # @unittest.skip
        def test_get_sample_and_noise_batch2(self):
            global predefined_noise
            arr_noise = [[[[i * 1.11]]] for i in range(1, 7)]
            arr_noise.reverse()
            predefined_noise = torch.Tensor(arr_noise)
            arr = [[[[[i]]]] for i in range(1, 7)]
            data_loader = torch.tensor(arr, dtype=torch.float)
            batch_size = 6
            match_size = 6
            device = 'cpu'
            ha = HungarianAlgoManagerDerived(data_loader, batch_size, match_size, device)
            for images, noises in ha.get_sample_and_noise_batch():
                self.assertEqual(len(images), 6)
                self.assertEqual(len(noises), 6)
                self.assertEqual(images[0], 1)
                self.assertEqual(images[1], 2)
                self.assertEqual(images[2], 3)
                self.assertEqual(images[3], 4)
                self.assertEqual(images[4], 5)
                self.assertEqual(images[5], 6)
                self.assertEqual(noises[0], 1.11)
                self.assertEqual(noises[1], 2.22)
                self.assertEqual(noises[2], 3.33)
                self.assertEqual(noises[3], 4.44)
                self.assertEqual(noises[4], 5.55)
                self.assertEqual(noises[5], 6.66)

        # @unittest.skip
        def test_get_sample_and_noise_batch3(self):
            global predefined_noise
            arr_noise = [[[[i * 1.11]]] for i in range(1, 7)]
            arr_noise.reverse()
            predefined_noise = torch.Tensor(arr_noise)
            arr = [[[[[i]]]] for i in range(1, 7)]
            data_loader = torch.tensor(arr, dtype=torch.float)
            batch_size = 3
            match_size = 6
            device = 'cpu'
            ha = HungarianAlgoManagerDerived(data_loader, batch_size, match_size, device)
            for b_idx, (images, noises) in enumerate(ha.get_sample_and_noise_batch()):
                if b_idx == 0:
                    self.assertEqual(len(images), 3)
                    self.assertEqual(len(noises), 3)
                    self.assertEqual(images[0], 1)
                    self.assertEqual(images[1], 2)
                    self.assertEqual(images[2], 3)
                    self.assertEqual(noises[0], 1.11)
                    self.assertEqual(noises[1], 2.22)
                    self.assertEqual(noises[2], 3.33)
                elif b_idx == 1:
                    self.assertEqual(len(images), 3)
                    self.assertEqual(len(noises), 3)
                    self.assertEqual(images[0], 4)
                    self.assertEqual(images[1], 5)
                    self.assertEqual(images[2], 6)
                    self.assertEqual(noises[0], 4.44)
                    self.assertEqual(noises[1], 5.55)
                    self.assertEqual(noises[2], 6.66)
                else:
                    self.fail(f"Unexpected batch size {b_idx}")
            # for

        # @unittest.skip
        def test_get_sample_and_noise_batch4(self):
            global predefined_noise_set_fn
            arr_noise1 = [[[[3.33]]], [[[2.22]]], [[[1.11]]]]
            arr_noise2 = [[[[6.66]]], [[[5.55]]], [[[4.44]]]]
            arr_noise1 = torch.Tensor(arr_noise1)
            arr_noise2 = torch.Tensor(arr_noise2)
            predefined_noise_set_fn = lambda img_set: arr_noise1 if img_set.flatten()[0] == 1. else arr_noise2
            arr = [[[[[i]]]] for i in range(1, 7)]
            data_loader = torch.tensor(arr, dtype=torch.float)
            batch_size = 6
            match_size = 3
            device = 'cpu'
            ha = HungarianAlgoManagerDerived(data_loader, batch_size, match_size, device)
            for images, noises in ha.get_sample_and_noise_batch():
                self.assertEqual(len(images), 6)
                self.assertEqual(len(noises), 6)
                self.assertEqual(images[0], 1)
                self.assertEqual(images[1], 2)
                self.assertEqual(images[2], 3)
                self.assertEqual(images[3], 4)
                self.assertEqual(images[4], 5)
                self.assertEqual(images[5], 6)
                self.assertEqual(noises[0], 1.11)
                self.assertEqual(noises[1], 2.22)
                self.assertEqual(noises[2], 3.33)
                self.assertEqual(noises[3], 4.44)
                self.assertEqual(noises[4], 5.55)
                self.assertEqual(noises[5], 6.66)

        # @unittest.skip
        def test_get_sample_and_noise_batch5(self):
            global predefined_noise_set_fn
            arr_noise1 = [[[[3.33]]], [[[2.22]]], [[[1.11]]]]
            arr_noise2 = [[[[6.66]]], [[[5.55]]], [[[4.44]]]]
            arr_noise1 = torch.Tensor(arr_noise1)
            arr_noise2 = torch.Tensor(arr_noise2)
            predefined_noise_set_fn = lambda img_set: arr_noise1 if img_set.flatten()[0] == 1. else arr_noise2
            arr = [[[[[i]]]] for i in range(1, 7)]
            data_loader = torch.tensor(arr, dtype=torch.float)
            batch_size = 2
            match_size = 3
            device = 'cpu'
            ha = HungarianAlgoManagerDerived(data_loader, batch_size, match_size, device)
            for b_idx, (images, noises) in enumerate(ha.get_sample_and_noise_batch()):
                if b_idx == 0:
                    self.assertEqual(len(images), 2)
                    self.assertEqual(len(noises), 2)
                    self.assertEqual(images[0], 1)
                    self.assertEqual(images[1], 2)
                    self.assertEqual(noises[0], 1.11)
                    self.assertEqual(noises[1], 2.22)
                elif b_idx == 1:
                    self.assertEqual(len(images), 2)
                    self.assertEqual(len(noises), 2)
                    self.assertEqual(images[0], 3)
                    self.assertEqual(images[1], 4)
                    self.assertEqual(noises[0], 3.33)
                    self.assertEqual(noises[1], 4.44)
                elif b_idx == 2:
                    self.assertEqual(len(images), 2)
                    self.assertEqual(len(noises), 2)
                    self.assertEqual(images[0], 5)
                    self.assertEqual(images[1], 6)
                    self.assertEqual(noises[0], 5.55)
                    self.assertEqual(noises[1], 6.66)
                else:
                    self.fail(f"Unexpected batch size {b_idx}")

        # @unittest.skip
        def test_get_sample_and_noise_batch6(self):
            global predefined_noise_set_fn
            arr_noise1 = [[[[3.33]]], [[[2.22]]], [[[1.11]]]]
            arr_noise2 = [[[[6.66]]], [[[5.55]]], [[[4.44]]]]
            arr_noise1 = torch.Tensor(arr_noise1)
            arr_noise2 = torch.Tensor(arr_noise2)
            predefined_noise_set_fn = lambda img_set: arr_noise1 if img_set.flatten()[0] == 1. else arr_noise2
            arr = [[[[[i]]]] for i in range(1, 7)]
            data_loader = torch.tensor(arr, dtype=torch.float)
            batch_size = 5
            match_size = 3
            device = 'cpu'
            ha = HungarianAlgoManagerDerived(data_loader, batch_size, match_size, device)
            for b_idx, (images, noises) in enumerate(ha.get_sample_and_noise_batch()):
                if b_idx == 0:
                    self.assertEqual(len(images), 5)
                    self.assertEqual(len(noises), 5)
                    self.assertEqual(images[0], 1)
                    self.assertEqual(images[1], 2)
                    self.assertEqual(images[2], 3)
                    self.assertEqual(images[3], 4)
                    self.assertEqual(images[4], 5)
                    self.assertEqual(noises[0], 1.11)
                    self.assertEqual(noises[1], 2.22)
                    self.assertEqual(noises[2], 3.33)
                    self.assertEqual(noises[3], 4.44)
                    self.assertEqual(noises[4], 5.55)
                elif b_idx == 1:
                    self.assertEqual(len(images), 1)
                    self.assertEqual(len(noises), 1)
                    self.assertEqual(images[0], 6)
                    self.assertEqual(noises[0], 6.66)
                else:
                    self.fail(f"Unexpected batch size {b_idx}")

    # class TestHungarianAlgoManager
    unittest.main()
