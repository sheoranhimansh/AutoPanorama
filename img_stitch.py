from typing import Union, List, Optional, Tuple
import math
import numpy as np
import scipy.sparse.csgraph as csgraph
import cv2
import matplotlib.pyplot as plt
import scipy.sparse.csr as csr
from stitch import _StitchImage


def fitting_rectangle(*points):
    t = float('inf')
    l = float('inf')
    r = float('-inf')
    b = float('-inf')
    for x, y in points:
        if x > r:
            r = x
        if y > b:
            b = y
        if y < t:
            t = y
        if x < l:
            l = x
    l = int(math.floor(l))
    t = int(math.floor(t))
    return (l, t), (int(math.ceil(r - l)), int(math.ceil(b - t)))


def im_pst(base, img, shift):
    print('---------------------------------------------------------------------------')
    print("Pasting...")
    h, w = img.shape[:2]
    x, y = shift
    dest_slice = np.s_[y:y + h, x:x + w]
    dest = base[dest_slice]
    dest = cv2.add(cv2.bitwise_and(dest, dest, mask=(255 - img[..., 3])), img)
    try:
        for i in range(y, y+h):
            for j in range(x, x+w):
                if list(base[i][j]) == [0, 0, 0, 0]:
                    base[i][j] = dest[i-y][j-x]
                else:
                    new = np.around((base[i][j] + dest[i-y][j-x]) )
                    print("Original", base[i][j])
                    base[i][j] = new
                    print(new, dest[i-y][j-x])
                    # base[i][j] = dest[i-y][j-x]
    except Exception as e:
        print("Failed@@@@@@@@@@@@@@@@@@@@")
        print(e)
    print('---------------------------------------------------------------------------')
    #base[dest_slice] = dest
    return


def colinfo(lab_image, mask=None):
    lab_image = lab_image.reshape(-1, lab_image.shape[-1]) if (mask is None) else lab_image[mask.nonzero()]
    return lab_image.mean(axis=0), lab_image.std(axis=0)


class ImageStitcher:
    def __init__(self, **kwargs):
        self._images = []
        self._matches = {}
        self.min_matches= 10
        self.ratio_threshold = 0.7
        self.centerp = None
        self.current_edge_matrix = None
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise NameError("Class '%s' does not have an attribute '%s'" % (
                    self.__class__.__name__, k))
            setattr(self, k, v)

    def add_image(self, image: Union[str, np.ndarray], fname: str=None):
        if isinstance(image, str):
            if fname is not None:
                fname = os.path.splitext(os.path.split(image)[1])[0]
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGBA)
        # 3 channels handling
        if img.shape[-1] == 3:
            img = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        image = _StitchImage(img, name=fname)
        # find features
        image.find_features()
        idx = len(self._images)
        self._images.append(image)

        for oidx, other in enumerate(self._images[:-1]):
            match = self.match_features(image, other)
            # Check matches
            if match is not None:
                self._matches[(idx, oidx)] = match

    @property
    def center(self) -> int:
        if self.centerp is None:
            self.centerp = self._find_center()
        return self.centerp

    @center.setter
    def center(self, val: int):
        self.centerp = val

    def stitch(self):
        """Main Stitch Func"""
        self.validate()
        print(self._images[self.center].name, 'considered image center')
        parents = csgraph.dijkstra(self._edge_matrix, directed=False, indices=self.center, return_predecessors=True)[1]
        print('Parent matrix:\n', parents)
        next_H = self.calculate_relative_homographies(parents)
        Hs = self.calculate_total_homographies(parents, next_H)
        all_new_corners = self.calculate_new_corners(Hs)
        base_shift, base_size = np.array(self.calculate_bounds(all_new_corners))
        order = self.calculate_draw_order(parents)
        canvas = np.zeros((base_size[1], base_size[0], 4), dtype=np.uint8)
        for j in order:
            image = self._images[j]
            new_corners = all_new_corners[j]
            H = Hs[j]
            shift, size = np.array(fitting_rectangle(*new_corners))
            dest_shift = shift - base_shift
            print('Post Transform of', image.name, 'is', *size)
            T = np.array([[1, 0, -shift[0]], [0, 1, -shift[1]], [0, 0, 1]])
            Ht = T.dot(H)
            print('Translated homography:\n', Ht)
            new_image = cv2.warpPerspective(image.image, Ht, tuple(size), flags=cv2.INTER_LINEAR,)
            try:
                im_pst(canvas, new_image, dest_shift)
            except Exception as e:
                print("Caught at func call")
                print(e)
        return canvas

    def validate(self):
        cc, groups = csgraph.connected_components(self._edge_matrix, directed=False)
        if cc != 1:
            most_common = np.bincount(groups).argmax()
            raise ValueError('Image(s) %s could not be stitched' % ','.join(
                self._images[img].name for img in np.where(groups != most_common)[0]
            ))

    def calculate_new_corners(self, Hs) -> List[np.array]:
        all_new_corners = []
        for image, H in zip(self._images, Hs):
            img = image.image
            corners = np.array([[0., 0.], [0., img.shape[1]], img.shape[:2], [img.shape[0], 0.],])
            new_corners = cv2.perspectiveTransform(corners.reshape(1, 4, 2), H)
            all_new_corners.append(new_corners[0])
        return all_new_corners

    def calculate_bounds(self, new_corners) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        all_corners = []
        for corners in new_corners:
            all_corners.extend(corners)
        corner, size = fitting_rectangle(*all_corners)
        print(len(all_corners), 'new corners to calculate bounds with', 'Center at:', (-corner[0], -corner[1]), 'Final Size:', size)
        return corner, size

    def calculate_draw_order(self, parents):
        order = csgraph.depth_first_order(csgraph.reconstruct_path(self._edge_matrix, parents, directed=False), self.center, return_predecessors=False)[::-1]
        print('Order to Draw:')
        strf = ''
        for i in order:
            strf += str(self._images[i].name) + ', '
        print(strf)
        return order

    def calculate_relative_homographies(self, parents):
        '''Calculate each homography src -> dst'''
        c = self.center
        next_H = []
        for src_idx, dst_idx in enumerate(parents):
            if dst_idx < 0 or src_idx == c:
                next_H.append(np.identity(3))
                continue
            matches = self._matches[(src_idx, dst_idx)] if (src_idx, dst_idx) in self._matches else self._matches[(dst_idx, src_idx)]
            swap = (src_idx, dst_idx) not in self._matches
            src, dst = self._images[src_idx], self._images[dst_idx]
            H = self._find_homography(src, dst, matches, swap=swap)
            next_H.append(H)
        return next_H

    def calculate_total_homographies(self, parents, next_H):
        cent = self.center
        total_H = [None] * len(parents)
        total_H[cent] = next_H[cent]
        path = []
        while any(i is None for i in total_H):
            path.append(next(n for n, i in enumerate(total_H) if i is None))
            while path:
                src_idx = path.pop()
                dst_idx = parents[src_idx]
                if cent == src_idx:
                    continue
                if total_H[dst_idx] is None:
                    path.extend((src_idx, dst_idx))
                else:
                    total_H[src_idx] = next_H[src_idx].dot(total_H[dst_idx])
        return total_H

    def _find_homography(self, src: _StitchImage, dst: _StitchImage,matches: List[cv2.DMatch], swap=False) -> np.ndarray:
        """Homography to Transform from src to dst"""
        print('Transforming', src.name, 'to', dst.name)
        if swap:
            src, dst = dst, src
        src_data = np.array([src.kp[i.queryIdx].pt for i in matches], dtype=np.float64).reshape(-1, 1, 2)
        dst_data = np.array([dst.kp[i.trainIdx].pt for i in matches], dtype=np.float64).reshape(-1, 1, 2)
        if swap:
            src_data, dst_data = dst_data, src_data
            src, dst = dst, src
        H, status = cv2.findHomography(src_data, dst_data, cv2.RANSAC, 2.)
        if status.sum() == 0:
            raise ValueError('Critical error finding homography - this should not happen')
        print('Homography for', src.name, '->', dst.name, H)
        return H

    def _find_center(self) -> int:
        shortest_path = csgraph.shortest_path(self._edge_matrix, directed=False,)
        center = np.argmin(shortest_path.max(axis=1))
        print('The center image is', self._images[center].name, '(index', center, ')')
        return center

    @property
    def _edge_matrix(self):
        if len(self._images) == 0:
            raise ValueError('Must have at least one image!')
        cur = self.current_edge_matrix
        if cur is not None and cur.shape[0] == len(self._images):
            return cur
        all_matches = list(self._matches)
        base = max(len(v) for v in self._matches.values()) + 1
        values = [base - len(self._matches[i]) for i in all_matches]
        self.current_edge_matrix = csr.csr_matrix((values, tuple(np.array(all_matches).T)), shape=(len(self._images), len(self._images)))
        print('New edge matrix:\n', self.current_edge_matrix.toarray())
        return self.current_edge_matrix

    def match_features(self, src: _StitchImage, dst: _StitchImage) -> Optional[List[cv2.DMatch]]:
        print('Matching features of', src.name, dst.name)
        matches = self.matcher.knnMatch(src.feat, dst.feat, k=2)
        good = [i for i, j in matches if i.distance < self.ratio_threshold * j.distance]
        print(len(matches), 'features matched', len(good), 'of which are good')
        if len(good) >= self.min_matches:
            print(src.name, '<=>', dst.name, 'score', len(good))
            return good
        return None
