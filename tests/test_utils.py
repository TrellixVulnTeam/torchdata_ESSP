from torchdata.utils import md5sum


def test_md5sum(tmpdir):
    file = (tmpdir / 'file.txt')
    file.write_text('This is a test.', encoding='utf-8')
    actual_md5 = md5sum(file, quiet=True)
    expected_md5 = '120ea8a25e5d487bf68b5f7096440019'
    assert actual_md5 == expected_md5
