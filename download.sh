
## output
wget  https://semanticnormsdata.s3.amazonaws.com/output.zip -O output.zip
unzip output.zip


## noun2prop2count
wget  https://semanticnormsdata.s3.amazonaws.com/noun2prop2count_ngram.p -O noun2prop2count_ngram.p
mv noun2prop2count_ngram.p models/ngram/

mkdir data && cd data

## concreteness data
wget https://semanticnormsdata.s3.amazonaws.com/concreteness.zip -O concreteness.zip
unzip concreteness.zip

## quantification data
wget https://semanticnormsdata.s3.amazonaws.com/quantification.zip -O quantification.zip
unzip quantification.zip


## datasets
mkdir datasets && cd datasets

## concept properties
mkdir concept_properties && cd concept_properties
wget https://semanticnormsdata.s3.amazonaws.com/concept_properties_no_img_embd.zip -O concept_properties_no_img_embd.zip
unzip concept_properties_no_img_embd.zip
cd ..

## feature norms
mkdir feature_norms && cd feature_norms
wget https://semanticnormsdata.s3.amazonaws.com/feature_norms_no_img_embd.zip -O feature_norms_no_img_embd.zip
unzip feature_norms_no_img_embd.zip
cd ..

## memory color
mkdir memory_color && cd memory_color
wget https://semanticnormsdata.s3.amazonaws.com/memory_color_no_img_embd.zip -O memory_color_no_img_embd.zip
unzip memory_color_no_img_embd.zip
cd ..

##uncomment to get image embeddings
# cd concept_properties
# wget https://semanticnormsdata.s3.amazonaws.com/concept_properties_img_embd.zip -O concept_properties_img_embd.zip
# unzip concept_properties_img_embd.zip
# cd ..

# cd feature_norms
# wget https://semanticnormsdata.s3.amazonaws.com/feature_norms_img_embd.zip -O feature_norms_img_embd.zip
# unzip feature_norms_img_embd.zip
# cd ..

# cd memory_color
# wget https://semanticnormsdata.s3.amazonaws.com/memory_color_img_embd.zip -O memory_color_img_embd.zip
# unzip memory_color_img_embd.zip
# cd ..