curl -s "https://api.osf.io/v2/nodes/ryjfm/files/osfstorage/" \
  | jq -r '.data[] | "\(.attributes.name) \(.id)"'
