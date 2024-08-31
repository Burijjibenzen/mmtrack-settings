from sot_interface import SingleObjectTracker

tracker = SingleObjectTracker("mixformer", "cuda:0")
tracker.set_param(input_path='demo/palace.mp4', screen_width=1920 / 2, screen_height=1080 / 2, output_path=None, rsl_w=1920, rsl_h=1080, slow=True)
tracker.fit()
tracker.track(show=True, thickness=3, bbox_output_path='bbox_output.txt', bbox_file='bbox_input.txt')