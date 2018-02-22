class Pipeline(object):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def process_pipeline(self, image):
        for process in self.pipeline:
            image = process(image)
        return image