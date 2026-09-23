"""Move RoboCasa's transient object XML to /tmp while preserving mesh paths.

Applies only to the isolated simulator copy under `.work`, never the submodule.
"""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
a = p.parse_args()
s = a.source.read_text()
if 'import tempfile\n' not in s:
    assert s.count('import time\n') == 1
    s = s.replace('import time\n', 'import time\nimport tempfile\n')
old_write = '''        time_str = str(time.time()).replace(".", "_")
        new_xml_path = os.path.join(folder, "{}_{}.xml".format(time_str, os.getpid()))
        f = open(new_xml_path, "w")
        f.write(xml_str)
        f.close()
'''
new_write = '''        # Model assets may be shared read-only. Resolve relative mesh paths
        # before writing the transient XML into a writable local directory.
        root = ET.fromstring(xml_str)
        for elem in root.findall(".//asset/mesh") + root.findall(".//asset/texture"):
            asset_file = elem.get("file")
            if asset_file and not os.path.isabs(asset_file):
                elem.set("file", os.path.abspath(os.path.join(folder, asset_file)))
        xml_str = ET.tostring(root, encoding="utf8").decode("utf8")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(xml_str)
            new_xml_path = f.name
'''
old_init = '''        super().__init__(
            fname=new_xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
            scale=scale,
        )

        # clean up xml - we don't need it anymore
        if os.path.exists(new_xml_path):
            os.remove(new_xml_path)
'''
new_init = '''        try:
            super().__init__(
                fname=new_xml_path,
                name=name,
                joints=[dict(type="free", damping="0.0005")],
                obj_type="all",
                duplicate_collision_geoms=False,
                scale=scale,
            )
        finally:
            os.remove(new_xml_path)
'''
for before, after in ((old_write, new_write), (old_init, new_init)):
    if before in s:
        s = s.replace(before, after)
    else:
        assert after in s, 'Unexpected RoboCasa source; refusing to patch'
a.source.write_text(s)
print('RoboCasa temporary XML path patched in isolated simulator copy')
