class Region:
    def __init__(self, name, radius_x, radius_y):
        self.name = name
        self.radius_x = radius_x
        self.radius_y = radius_y

    def __str__(self):
        return (self.name, self.radius_x, self.radius_y).__str__()


def get_regions(region_file_path):
    cfg = open(region_file_path, 'r')
    regions = dict()
    """
    region: (City Name, radius_x, radius_y)
    regions: [region1, region2, ...]
    """
    for line in cfg.readlines():
        params = line.strip().split(' ')
        if len(params) == 3:
            city, radius_x, radius_y = params
        else:  # 4
            city, city_extend, radius_x, radius_y = params
            city += " " + city_extend
        region = Region(city, int(radius_x), int(radius_y))
        # regions.append(region)
        regions[region.name] = region
        # print(region)
    return regions
