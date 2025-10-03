# PixelCharacter

kaggle datasets download -d volodymyrpivoshenko/pixel-characters-dataset

How to run app:

```streamlit run main.py```


```
Table Users {
  user_id varchar [primary key] 
  username varchar
  password varchar
  date_created datetime
}

Table User_Details {
  id varchar [primary key]
  user_id varchar 
  email varchar
  date_updated datetime
  
}

Table Image_Details {
  image_id integer [primary key]
  user_id integer 
  name varchar
  image_path varchar
}

Table Image_Binary {
  image_id integer [primary key]
  raw varbinary
}

Table Blend {
  image_id  integer [primary key]
  source_image1 integer
  source_image2 integer
  blend_ratio  integer
  
}

Ref: "User_Details"."user_id" < "Image_Details"."user_id"

Ref: "Image_Details"."image_id" < "Blend"."image_id"

Ref: "Image_Binary"."image_id" < "Image_Details"."image_id"

Ref: "User_Details"."user_id" < "Users"."username"

Ref: "Blend"."source_image1" < "Image_Details"."image_id"

Ref: "Blend"."source_image2" < "Image_Details"."image_id"
```