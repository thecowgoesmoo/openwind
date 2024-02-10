#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from datetime import datetime

try:
    new_version = sys.argv[1] # Get the new version number
except IndexError:
    print("ERROR: You forgot to give the new version number as argument")
    print("Exiting...")
    sys.exit(0)
current_date = datetime.today().strftime('%Y-%m-%d')


# Open latest_changelog.md and put it in a list
with open("docs/latest_changelog.md", "r") as latest_changelog:
    latest_list = latest_changelog.readlines()
    # Insert release title, date and link to release
    latest_list.insert(0, "## [%s]""(https://gitlab.inria.fr/openwind/openwind/-/releases/v%s) - %s \n\n"
                       %(new_version, new_version, current_date))
    latest_list.append("\n\n")
    latest_str = "".join(latest_list)
    latest_changelog.close()
# Open global CHANGELOG.md and insert latest changelog at the beginning (line 5)
with open("CHANGELOG.md", "r") as current_changelog:
    changelog_list = current_changelog.readlines()
    current_changelog.close()

changelog_list.insert(5, latest_str)

# Write to CHANGELOG.md the updated file
with open("CHANGELOG.md", "w") as update_changelog:
    update_changelog.writelines("".join(changelog_list))
    update_changelog.close()
