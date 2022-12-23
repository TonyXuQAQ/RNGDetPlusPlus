import math


class Point(object):
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def sub(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def add(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def scale(self, f):
        return Point(self.x * f, self.y * f)

    def magnitude(self):  # |vector|
        return math.sqrt(self.x * self.x + self.y * self.y)

    def angle_to(self, other):
        if self.magnitude() == 0 or other.magnitude() == 0:
            return 0
        s = (self.x * other.x + self.y * other.y) / self.magnitude() / other.magnitude()
        if abs(s) > 1: s = s / abs(s)
        angle = math.acos(s)
        if angle > math.pi:
            return 2 * math.pi - angle
        else:
            return angle

    def signed_angle(self, other):
        return math.atan2(other.y, other.x) - math.atan2(self.y, self.x)

    def bounds(self):
        return Rectangle(self, self)

    def dot(self, point):
        return self.x * point.x + self.y * point.y

    def __repr__(self):
        return 'Point({}, {})'.format(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.x, self.y))


class FPoint(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def sub(self, other):
        return FPoint(self.x - other.x, self.y - other.y)

    def add(self, other):
        return FPoint(self.x + other.x, self.y + other.y)

    def scale(self, f):
        return FPoint(self.x * f, self.y * f)

    def magnitude(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def angle_to(self, other):
        if self.magnitude() == 0 or other.magnitude() == 0:
            return 0
        s = (self.x * other.x + self.y * other.y) / self.magnitude() / other.magnitude()
        if abs(s) > 1: s = s / abs(s)
        angle = math.acos(s)
        if angle > math.pi:
            return 2 * math.pi - angle
        else:
            return angle

    def signed_angle(self, other):
        return math.atan2(other.y, other.x) - math.atan2(self.y, self.x)

    def dot(self, point):
        return self.x * point.x + self.y * point.y

    def __repr__(self):
        return 'FPoint({}, {})'.format(self.x, self.y)


class Segment(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def length(self):
        return self.start.distance(self.end)

    def project_factor(self, point):
        l = self.length()
        if l == 0:
            return 0
        t = point.sub(self.start).dot(self.end.sub(self.start)) / l
        t = max(0, min(l, t))
        return t

    def project(self, point):
        t = self.project_factor(point)
        return self.point_at_factor(t)

    def point_at_factor(self, t):
        l = self.length()
        if l == 0:
            return self.start
        return self.start.add(self.end.sub(self.start).scale(t / l))

    def distance(self, point):
        p = self.project(point)
        return p.distance(point)

    def intersection(self, other):
        d1 = self.vector()
        d2 = other.vector()
        d12 = other.start.sub(self.start)

        den = d1.y * d2.x - d1.x * d2.y
        u1 = d1.x * d12.y - d1.y * d12.x
        u2 = d2.x * d12.y - d2.y * d12.x

        if den == 0:
            # collinear
            if u1 == 0 and u2 == 0:
                return self.start
            else:
                return None

        if float(u1) / den < 0 or float(u1) / den > 1 or float(u2) / den < 0 or float(u2) / den > 1:
            return None

        return self.point_at_factor(float(u2) / den * self.length())

    def vector(self):
        return self.end.sub(self.start)

    def bounds(self):
        return self.start.bounds().extend(self.end)

    def extend(self, amount):
        v = self.vector()
        v = v.scale(amount / v.magnitude())
        return Segment(
            self.start.sub(v),
            self.end.add(v)
        )

    def __repr__(self):
        return 'Segment({}, {})'.format(self.start, self.end)


class Rectangle(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def lengths(self):
        return Point(self.end.x - self.start.x, self.end.y - self.start.y)

    def clip(self, point):
        npoint = Point(point.x, point.y)
        if npoint.x < self.start.x:
            npoint.x = self.start.x
        elif npoint.x >= self.end.x:
            npoint.x = self.end.x - 1
        if npoint.y < self.start.y:
            npoint.y = self.start.y
        elif npoint.y >= self.end.y:
            npoint.y = self.end.y - 1
        return npoint

    def clip_rect(self, r):
        return Rectangle(self.clip(r.start), self.clip(r.end))

    def add_tol(self, tol):
        return Rectangle(
            self.start.sub(Point(tol, tol)),
            self.end.add(Point(tol, tol))
        )

    def contains(self, point):
        return point.x >= self.start.x and point.x < self.end.x and point.y >= self.start.y and point.y < self.end.y

    def extend(self, point):
        return Rectangle(
            Point(min(self.start.x, point.x), min(self.start.y, point.y)),
            Point(max(self.end.x, point.x), max(self.end.y, point.y))
        )

    def intersects(self, other):
        return self.end.y >= other.start.y and other.end.y >= self.start.y and self.end.x >= other.start.x and other.end.x >= self.start.x

    def translate(self, x, y):
        return Rectangle(
            self.start.add(Point(x, y)),
            self.end.add(Point(x, y))
        )
    
    def at_bounds(self, point):
        return (point.x == self.start.x or point.y == self.start.y or point.x == self.end.x-1 or point.y == self.end.y-1) \
                and self.contains(point)


def draw_line(start, end, lengths):
    # followX indicates whether to move along x or y coordinates
    followX = abs(end.y - start.y) <= abs(end.x - start.x)
    if followX:
        x0 = start.x
        x1 = end.x
        y0 = start.y
        y1 = end.y
    else:
        x0 = start.y
        x1 = end.y
        y0 = start.x
        y1 = end.x

    delta = Point(abs(x1 - x0), abs(y1 - y0))
    current_error = 0

    if x0 < x1:
        xstep = 1
    else:
        xstep = -1

    if y0 < y1:
        ystep = 1
    else:
        ystep = -1

    points = []

    def add_point(p):
        if p.x >= 0 and p.x < lengths.x and p.y >= 0 and p.y < lengths.y:
            points.append(p)

    x = x0
    y = y0

    while x != x1 + xstep:
        if followX:
            add_point(Point(x, y))
        else:
            add_point(Point(y, x))

        x += xstep
        current_error += delta.y
        if current_error >= delta.x:
            y += ystep
            current_error -= delta.x

    return points


def vector_from_angle(angle, length):
    return Point(math.cos(angle) * length, math.sin(angle) * length)
