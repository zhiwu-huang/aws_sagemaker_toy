export IMG='number_5.png'
export URL=`chalice url`

# Download test image
echo -e "\n********************** Downloading test image **********************\n"
curl -o $IMG https://blog.otoro.net/assets/20160401/png/mnist_output_10.png

echo -e "\n********************** POST request to Lambda **********************\n"
(echo -n '{"data": "'; base64 $IMG; echo '"}') |
curl -H "Content-Type: application/json" -d @- $URL | json_pp

# Remove test image
[ -e $IMG ] && rm $IMG
