import os

import lsml as slls

base = os.path.dirname(level_set_learn.__file__)

s = "level_set_machine_learning"

contents = """\
{module}
{dashes}

.. automodule:: {module}
    :members:
"""

index = """\
SLLS API
--------

.. toctree::
    :maxdepth: 1
"""

index_item = " "*4 + "{item}\n"


for dirpath, dirnames, filenames in os.walk(base):
    for fname in filenames:
        if fname.startswith("_") or \
            not fname.endswith(".py") or \
            dirpath.endswith('tests'):
            continue

        path = dirpath[dirpath.index(s) + len(s) + 1 :]

        if path != "" and not os.path.exists(path):
            os.makedirs(path)

        name = os.path.splitext(fname)[0]
        
        module = os.path.join(s, path, name).replace("/", ".")
        dashes = "-" * len(module)

        rst_name = name+".rst"

        with open(os.path.join(path, rst_name), 'w') as f:
            c = contents.format(dashes=dashes, module=module)
            f.write(c)

        index += index_item.format(item=os.path.join(path, name))

with open("index.rst", 'w') as f:
    f.write(index)

