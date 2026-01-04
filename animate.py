#!/usr/bin/env python3

"""
make animation using the .npz file

to do: check if the CE overlay works
test with more extremes
- large eccentricity
-lower masses (below 1 msun)
-very very large radii
    - large diffe4renes in radii for both stars
"""

import math
from pathlib import Path
import numpy as np
import pygame
from pygame import gfxdraw
from PIL import Image
import argparse

BASE_DIR = Path(__file__).parent
STELLAR_IMG_DIR = BASE_DIR / "Images"
FRAMES_FILE = BASE_DIR / "frames_data.npz"
BACKGROUND_PATH = STELLAR_IMG_DIR / "Background.png"
CE_OVERLAY_PATH = STELLAR_IMG_DIR / "common_envelope.png"


SCREEN_SIZE = (1000, 800)
FPS_PLAYBACK = 50
PIXELS_AT_REF = 100
PIXELS_AT_REF_LIN = 300
USE_LOG_SCALING = False  # True = log scaling, False = linear scaling
USE_TULIPS_COLOR = False  # True = colored circles, False = stellar images
#the true false things are now through the terminal commmand to run the animation 

#helpers
def log_scaled_radius(radius_rsun, pixels_at_ref=100):
    radius_rsun = max(radius_rsun, 1e-6)
    log_scale = math.log10(radius_rsun + 1.0)
    reference_log = math.log10(100.0 + 1.0)
    return (log_scale / reference_log) * pixels_at_ref

def linear_scaled_radius(radius_rsun, pixels_at_ref_lin=300):
    radius_rsun = max(radius_rsun, 0.0)
    return  (radius_rsun / 100.0) * pixels_at_ref_lin


def scaled_radius(radius_rsun, pixels_at_ref=100):
    if USE_LOG_SCALING:
        return log_scaled_radius(radius_rsun, pixels_at_ref)
    else:
        return linear_scaled_radius(radius_rsun, pixels_at_ref)

def make_antialiased_circle(radius_px, color):
   
    r = max(int(radius_px), 2)
    size = 2 * r + 2

    surf = pygame.Surface((size, size), pygame.SRCALPHA)

    gfxdraw.filled_circle(surf, r+1, r+1, r, (*color, 255))
    gfxdraw.aacircle(surf, r+1, r+1, r, (*color, 255))

    return surf

class PygameAnimator:
    def __init__(self, frames_file, image_dir=STELLAR_IMG_DIR, background_path=BACKGROUND_PATH, ce_path=CE_OVERLAY_PATH):
        self.frames_file = Path(frames_file)
        self.image_dir = Path(image_dir)
        self.background_path = Path(background_path)
        self.ce_path = Path(ce_path)
        self.frames = []
        self.load_frames()
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("COMPAS Evolution Animation (pygame)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.bold_font = pygame.font.SysFont("Arial", 18, bold=True)
        self.img_cache = {}
        self.ce_img = None
        self.mt_img_path = self.image_dir / "mass_transfer_stream.png" 
        self.mt_img = None
        if self.mt_img_path.exists():
            pil = Image.open(self.mt_img_path).convert("RGBA")
            self.mt_img = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode).convert_alpha()
        if self.ce_path.exists():
            pil = Image.open(self.ce_path).convert("RGBA")
            mode = pil.mode
            size = pil.size
            data = pil.tobytes()
            self.ce_img = pygame.image.fromstring(data, size, mode).convert_alpha()
        self.bg_surf = None
        if self.background_path.exists():
            pil = Image.open(self.background_path).convert("RGBA")
            pil = pil.resize(SCREEN_SIZE, Image.Resampling.LANCZOS)
            self.bg_surf = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode).convert_alpha()
        self.stellar_list_img_path = self.image_dir / "Stellar_Type_List.png"
        self.stellar_list_img = None
        if self.stellar_list_img_path.exists():
            pil = Image.open(self.stellar_list_img_path).convert("RGBA")
            self.stellar_list_img = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode).convert_alpha()
        self.supernova_img_path = self.image_dir / "supernova.png"
        self.supernova_img = None
        if self.supernova_img_path.exists():
            pil = Image.open(self.supernova_img_path).convert("RGBA")
            self.supernova_img = pygame.image.fromstring(pil.tobytes(), pil.size, pil.mode).convert_alpha()

        self.bh_birth_frame1 = None  # first frame where star1 becomes BH
        self.bh_birth_frame2 = None  # first frame where star2 becomes BH

        self.determine_scale()

    def determine_scale(self):
        # estimate max extent from frames for visual mapping
        max_extent = 1.0
        for f in self.frames:
            rmax = max(f['Radius(1)'], f['Radius(2)'])
            a = f['SemiMajorAxis'] * (1 + f['Eccentricity'])
            max_extent = max(max_extent, rmax*4 + a)
       
        ref_px = scaled_radius(max_extent)
        if ref_px <= 0:
            ref_px = 100
        target_half = 0.6 * min(SCREEN_SIZE)/2.0
        self.scale = target_half / ref_px

    def load_frames(self):
        dat = np.load(str(self.frames_file), allow_pickle=True)
        raw = dat['frames']
        self.frames = [dict(x) for x in raw.tolist()]
        print(f"Loaded {len(self.frames)} frames.")

    def get_star_surface(self, stype_name, radius_rsun):
        key = (stype_name, int(round(radius_rsun)))
        if key in self.img_cache:
            return self.img_cache[key]
        img_path = self.image_dir / f"{stype_name}.png"
        if not img_path.exists():
            # fallback
            r_px = max(int(scaled_radius(radius_rsun)), 10) 
             # minimum radius 10 px
            surf = pygame.Surface((2*r_px, 2*r_px), pygame.SRCALPHA)
            gfxdraw.filled_circle(surf, r_px, r_px, r_px, (200,200,255,255))
            self.img_cache[key] = surf
            return surf
        # load with PIL first for scaling quality then convert to pygame
        #it looks really wierd and pixelated without this
        pil = Image.open(img_path).convert("RGBA")
        # scale relative to radius
        r_px = max(int(scaled_radius(radius_rsun)), 10) 
         # minimum radius 10 px
        w,h = pil.size
        scale_factor = (2 * r_px) / h
        new_w = max(4, int(w * scale_factor))
        new_h = max(4, int(h * scale_factor))
        pil_resized = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        surf = pygame.image.fromstring(pil_resized.tobytes(), pil_resized.size, pil_resized.mode).convert_alpha()
        self.img_cache[key] = surf
        return surf
    
    def get_colored_circle_surface(self, radius_rsun, rgb, u=0.8):
       
        r_px = max(int(scaled_radius(radius_rsun)), 10)
        size = 2 * r_px

        surf = pygame.Surface((size, size), pygame.SRCALPHA)

        cx = cy = r_px
        R = float(r_px)

        base_r, base_g, base_b = rgb

        for y in range(size):
            dy = y - cy
            for x in range(size):
                dx = x - cx
                r = math.hypot(dx, dy)

                if r <= R:
                    # mu = cos(theta) = sqrt(1 - (r/R)^2)
                    #from online
                    mu = math.sqrt(1.0 - (r / R) ** 2)

                    # the darker outer edge
                    #this is from a limb darkening equation i got online but have no clue if its the correct way to use it
                    #it looks fine tho so ill fix / double check later
                    intensity = 1.0 - u * (1.0 - mu)

                    col = (
                        int(base_r * intensity),
                        int(base_g * intensity),
                        int(base_b * intensity),
                        255
                    )

                    surf.set_at((x, y), col)

        return surf




    def draw_mass_transfer_hourglass(self, surface, start_xy, end_xy, rstart_px, rend_px):
        x1,y1 = start_xy
        x4,y4 = end_xy
        vec = np.array([x4-x1, y4-y1], dtype=float)
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return
        unit = vec/length
        perp = np.array([-unit[1], unit[0]])
        width_start = 0.8*(2*rstart_px)
        width_end = 0.8*(2*rend_px)
        p1 = np.array([x1,y1]) + (width_start/2.0)*perp
        p2 = np.array([x1,y1]) - (width_start/2.0)*perp
        p3 = np.array([x4,y4]) - (width_end/2.0)*perp
        p4 = np.array([x4,y4]) + (width_end/2.0)*perp
        neck_center = np.array([x1,y1]) + vec*0.5
        neck_width = 0.4*(width_start + width_end)/2.0
        p_mid_upper = neck_center + (neck_width/2.0)*perp
        p_mid_lower = neck_center - (neck_width/2.0)*perp

        left_curve = cubic_bezier_array(p1, p1 + vec*0.35 - perp*(width_start*0.25), p_mid_upper + perp*(neck_width*0.15), p_mid_upper, n=30)
        right_curve = cubic_bezier_array(p2, p2 + vec*0.35 + perp*(width_start*0.25), p_mid_lower - perp*(neck_width*0.15), p_mid_lower, n=30)
        end_upper_line = np.linspace(p_mid_upper, p4, 30)
        end_lower_line = np.linspace(p_mid_lower, p3, 30)[::-1]

        polygon = np.vstack([left_curve, end_upper_line, end_lower_line, right_curve[::-1]])
        poly_pts = [(int(px), int(py)) for px,py in polygon]
        # colo: a pale buttercupish yellow (looks brighter on black backgrouhnd)
        col = (255, 251, 207, 120)
        s = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(s, poly_pts, col)
        surface.blit(s, (0,0))

    def run(self):
        running = True
        frame_idx = 0
        n_frames = len(self.frames)
        
        for idx, f in enumerate(self.frames):
            if self.bh_birth_frame1 is None and f['stypeName1'] == 'BH':
                self.bh_birth_frame1 = idx
            if self.bh_birth_frame2 is None and f['stypeName2'] == 'BH':
                self.bh_birth_frame2 = idx
            if self.bh_birth_frame1 is not None and self.bh_birth_frame2 is not None:
                break

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.clock.tick(FPS_PLAYBACK)
            # draw background
            if self.bg_surf:
                self.screen.blit(self.bg_surf, (0,0))
            else:
                self.screen.fill((0,0,0))

            f = self.frames[frame_idx]
            #change 200 to a lower value for slower orbit
            #latrger value foe faster
            orbit_t = (frame_idx % 200) / 200.0  
            a = f['SemiMajorAxis']
            e = f['Eccentricity']
            b = a * math.sqrt(max(0.0, 1 - e**2))
            theta = 2*math.pi*orbit_t
            x1_phys = a * math.cos(theta) - a*e
            y1_phys = b * math.sin(theta)
            dist1 = math.sqrt(x1_phys**2 + y1_phys**2)
            # compute scale from physical distances->log(px)
            if dist1 > 0:
                scale_dist = scaled_radius(dist1) / dist1
                # if using linear, multiply by 3
                if not USE_LOG_SCALING:
                    scale_dist *= 3.0
            else:
                scale_dist = 0.0
            x1 = x1_phys * scale_dist
            y1 = y1_phys * scale_dist

            mass_ratio = (f['Mass(1)'] / f['Mass(2)']) if (f['Mass(2)'] > 0) else 1.0
            x2 = -x1 * mass_ratio
            y2 = -y1 * mass_ratio

            # Map scaled coords to screen coords
            sx1, sy1 = int(SCREEN_SIZE[0]//2 + x1*self.scale), int(SCREEN_SIZE[1]//2 - y1*self.scale)
            sx2, sy2 = int(SCREEN_SIZE[0]//2 + x2*self.scale), int(SCREEN_SIZE[1]//2 - y2*self.scale)

            if 'mass transfer' in f.get('eventString','').lower() and self.mt_img:
                dx = sx2 - sx1
                dy = sy2 - sy1
                dist = math.hypot(dx, dy)
                angle = math.degrees(math.atan2(dy, dx))
                #determine smaller star radius in SCREEN pixels
                r1_px = (surf1.get_width() // 2)
                r2_px = (surf2.get_width() // 2)
                r_small_px = min(r1_px, r2_px)

                stream_length = max(int(dist), 10)

                stream_max_width = 2 * r_small_px

                # preserve original aspect ratio, then clamp width
                orig_w, orig_h = self.mt_img.get_size()
                aspect = orig_h / orig_w

                stream_height = int(stream_length * aspect)

                # cap the height so it never exceeds star diameter
                stream_height = min(stream_height, stream_max_width)

                mt_scaled = pygame.transform.smoothscale(
                    self.mt_img,
                    (stream_length, stream_height)
                )

                mt_rotated = pygame.transform.rotate(mt_scaled, -angle) 
                 # pygame rotates counter-clockwise
                # center between stars
                midx = (sx1+sx2)//2 - mt_rotated.get_width()//2
                midy = (sy1+sy2)//2 - mt_rotated.get_height()//2
                temp = mt_rotated.copy()
                temp.set_alpha(150) 
                self.screen.blit(temp, (midx, midy))



            if USE_TULIPS_COLOR:
                surf1 = self.get_colored_circle_surface(
                    f['Radius(1)'], f['RGB1']
                )
                surf2 = self.get_colored_circle_surface(
                    f['Radius(2)'], f['RGB2']
                )
            else:
                surf1 = self.get_star_surface(f['stypeName1'], f['Radius(1)'])
                surf2 = self.get_star_surface(f['stypeName2'], f['Radius(2)'])

            # Override circles with images for compact objects, since idt thats accurate for a NS but not sure
            if USE_TULIPS_COLOR:
                if f['stypeName1'] in ('BH', 'NS', 'WD'):
                    surf1 = self.get_star_surface(f['stypeName1'], f['Radius(1)'])
                if f['stypeName2'] in ('BH', 'NS', 'WD'):
                    surf2 = self.get_star_surface(f['stypeName2'], f['Radius(2)'])


            r1w, r1h = surf1.get_size()
            r2w, r2h = surf2.get_size()
            self.screen.blit(surf1, (sx1 - r1w//2, sy1 - r1h//2))
            self.screen.blit(surf2, (sx2 - r2w//2, sy2 - r2h//2))

            # Overlay supernova on newly formed black holes
            #should prob adjust size at some point
            if self.supernova_img:
                # star 1
                if self.bh_birth_frame1 is not None and self.bh_birth_frame1 <= frame_idx < self.bh_birth_frame1 + 50:
                    sn_w, sn_h = self.supernova_img.get_size()
                    scale_factor = max(int(scaled_radius(f['Radius(1)'])*2 / sn_h), 1)
                    # apply 50% size
                    sn_scaled = pygame.transform.smoothscale(self.supernova_img, (int(sn_w*scale_factor*0.5), int(sn_h*scale_factor*0.5)))
                    self.screen.blit(sn_scaled, (sx1 - sn_scaled.get_width()//2, sy1 - sn_scaled.get_height()//2))
                # star 2
                if self.bh_birth_frame2 is not None and self.bh_birth_frame2 <= frame_idx < self.bh_birth_frame2 + 50:
                    sn_w, sn_h = self.supernova_img.get_size()
                    scale_factor = max(int(scaled_radius(f['Radius(2)'])*2 / sn_h), 1)
                    # apply 50% size
                    sn_scaled = pygame.transform.smoothscale(self.supernova_img, (int(sn_w*scale_factor*0.5), int(sn_h*scale_factor*0.5)))
                    self.screen.blit(sn_scaled, (sx2 - sn_scaled.get_width()//2, sy2 - sn_scaled.get_height()//2))


            txt_color = (255, 255, 255)  # white
            txt1_surf = self.bold_font.render("M1", True, txt_color)
            txt2_surf = self.bold_font.render("M2", True, txt_color)

            self.screen.blit(txt1_surf, (sx1 + r1w//2 + 5, sy1 - txt1_surf.get_height()//2))
            self.screen.blit(txt2_surf, (sx2 + r2w//2 + 5, sy2 - txt2_surf.get_height()//2))



            t_max = max(f_['Time'] for f_ in self.frames)  # maximum simulation time in Myr

            # determine rounding and tick interval
            if t_max <= 10:
                t_max_round = math.ceil(t_max)  # nearest 1 Myr
                tick_interval = 1
                small_tick_interval = 0.5
            elif t_max <= 100:
                t_max_round = math.ceil(t_max / 5) * 5  # nearest 5 Myr
                tick_interval = 5
                small_tick_interval = 1
            else:
                t_max_round = math.ceil(t_max / 10) * 10  # >=100 Myr, round 10 Myr
                tick_interval = 10
                small_tick_interval = 2
            text_block_max_width = 300 
            buffer = 20                  
            # space between text and timeline
            timeline_x_start = 30 + text_block_max_width + buffer
            timeline_x_end = SCREEN_SIZE[0] - 30                
            timeline_length_px = timeline_x_end - timeline_x_start
            timeline_y = 50  # top of screen

            pygame.draw.line(self.screen, (255,255,255), (timeline_x_start, timeline_y), (timeline_x_end, timeline_y), 2)

            # major ticks
            num_major_ticks = int(t_max_round / tick_interval) + 1
            for i in range(num_major_ticks):
                t = i * tick_interval
                x = int(timeline_x_start + (t / t_max_round) * timeline_length_px)
                pygame.draw.line(self.screen, (255,255,255), (x, timeline_y-6), (x, timeline_y+6), 2)
                # label
                label_surf = self.font.render(f"{int(t)}", True, (255,255,255))
                self.screen.blit(label_surf, (x - label_surf.get_width()//2, timeline_y - 25))

            # minor ticks
            num_minor_ticks = int(t_max_round / small_tick_interval) + 1
            for i in range(num_minor_ticks):
                t = i * small_tick_interval
                if t % tick_interval == 0:
                    continue 
                x = int(timeline_x_start + (t / t_max_round) * timeline_length_px)
                pygame.draw.line(self.screen, (255,255,255), (x, timeline_y-4), (x, timeline_y+4), 1)

            # current frame time
            t_now = f['Time']
            x_dot = int(timeline_x_start + (t_now / t_max_round) * timeline_length_px)
            pygame.draw.circle(self.screen, (255,0,0), (x_dot, timeline_y), 6)

            label_surf = self.font.render("Time in Myr", True, (255,255,255))
            self.screen.blit(label_surf, (timeline_x_start + timeline_length_px // 2 - label_surf.get_width() // 2,timeline_y + 10))  # 10 px below timeline

            # Common envelope overlay
            if self.ce_img and 'common envelope' in f.get('eventString','').lower():
                dx = sx2 - sx1
                dy = sy2 - sy1
                dist = math.hypot(dx, dy)
                w = int(dist*1.8)
                if w < 10:
                    w = 10
                h = int(w * (self.ce_img.get_height()/self.ce_img.get_width()))
                ce_scaled = pygame.transform.smoothscale(self.ce_img, (w,h))
                midx = (sx1+sx2)//2 - w//2
                midy = (sy1+sy2)//2 - h//2
                temp = ce_scaled.copy()
                temp.set_alpha(100)
                self.screen.blit(temp, (midx, midy))

            lines = [
                f"t = {f['Time']:.2f} Myr",
                f"a = {f['SemiMajorAxis']:.2f} R_sun, e = {f['Eccentricity']:.2f}",
                f"M1 = {f['Mass(1)']:.2f} M_sun, M2 = {f['Mass(2)']:.2f} M_sun",
                f"R1 = {f['Radius(1)']:.2f} R_sun, R2 = {f['Radius(2)']:.2f} R_sun", 
                f"{f.get('eventString','')}"
            ]
            x0 = 30
            y0 = 30
            for i,l in enumerate(lines):
                txt_surf = self.font.render(l, True, (255,255,255))
                self.screen.blit(txt_surf, (x0, y0 + i*20))


            if (not USE_TULIPS_COLOR) and self.stellar_list_img:
                orig_w, orig_h = self.stellar_list_img.get_size()
                new_w, new_h = orig_w // 2, orig_h // 2
                img_scaled = pygame.transform.smoothscale(self.stellar_list_img, (new_w, new_h))

                img_x = 30
                img_y = 30 + len(lines)*20 + 10
                self.screen.blit(img_scaled, (img_x, img_y))

            # Draw reference scale at bottom
            if USE_LOG_SCALING:
                major_ticks = [0, 10, 100, 1000]

                minor_ticks = []
                for i in range(len(major_ticks)-1):
                    start = major_ticks[i]
                    end = major_ticks[i+1]
                    decade = end // 10
                    for mult in [2,4,6,8]:
                        tick = start + mult * decade
                        if tick < end:
                            minor_ticks.append(tick)

                scale_bar_ticks = sorted(set(major_ticks + minor_ticks))

                scale_bar_length_rsun = major_ticks[-1] 

            else:
                # linear scaling
                scale_bar_length_rsun = 1000.0      
                tick_interval_rsun = 100
                scale_bar_ticks = list(range(0, int(scale_bar_length_rsun)+1, tick_interval_rsun))
                major_ticks = scale_bar_ticks[::2] 

            scale_px = scaled_radius(scale_bar_length_rsun) * self.scale

            margin = 40
            x_start = SCREEN_SIZE[0]//2 - scale_px//2
            y_start = SCREEN_SIZE[1] - margin
            label_surf = self.font.render("Distance Scaling in Solar Radii", True, (255,255,255))
            self.screen.blit(label_surf, (SCREEN_SIZE[0]//2 - label_surf.get_width()//2, y_start - 25))

            pygame.draw.line(self.screen, (255,255,255), (x_start, y_start), (x_start + scale_px, y_start), 2)

            for r in scale_bar_ticks:
                x_tick = x_start + scaled_radius(r) * self.scale
                tick_height = 6
                pygame.draw.line(self.screen, (255,255,255), (x_tick, y_start - tick_height//2), (x_tick, y_start + tick_height//2), 2)
                if r in major_ticks:
                    label_surf = self.font.render(str(int(r)), True, (255,255,255))
                    self.screen.blit(label_surf, (x_tick - label_surf.get_width()//2, y_start + tick_height + 2))

            pygame.display.flip()

            frame_idx += 1
            if frame_idx >= n_frames:
                frame_idx = 0  # loop

        pygame.quit()

import argparse

def parse_cmd_arguments():
   
    parser = argparse.ArgumentParser(description="Process scaling and image settings.")

    parser.add_argument(
        "--scaling", 
        choices=["log", "linear"], 
        help="The type of scaling to apply (log or linear)."
    )

    parser.add_argument(
        "--images", 
        choices=["tulips", "default"], 
        help="The set of images to use (tulips or default)."
    )

    return parser.parse_args()

#not sure whether this actually makes it more convenient
#i was told this is a better practice than changing it in the actual code so ill leave it for now


def main():
    global USE_LOG_SCALING
    global USE_TULIPS_COLOR

    if not FRAMES_FILE.exists():
        raise FileNotFoundError(
            "frames_data.npz not found. Run compas_preprocess.py first."
        )
    args = parse_cmd_arguments()
    if args.scaling == "log":
        USE_LOG_SCALING = True
    elif args.scaling == "linear":
        USE_LOG_SCALING = False

    if args.images == "tulips":
        USE_TULIPS_COLOR = True
    elif args.images == "default":
        USE_TULIPS_COLOR = False

    print(f"scaling {args.scaling}, images {args.images}")
    animator = PygameAnimator(FRAMES_FILE)
    animator.run()


if __name__ == "__main__":
    main()
