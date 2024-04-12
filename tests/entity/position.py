from casrl.entity.position import Position


def test_update():
    pos = Position(0, 0)
    pos.update(1, 2)
    assert pos.x == 1
    assert pos.y == 2
