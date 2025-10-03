CREATE TABLE Users (
  user_id integer PRIMARY KEY,
  username varchar(255),
  password varchar(255),
  date_created datetime,
  date_updated datetime
);

CREATE TABLE User_Details (
  id integer PRIMARY KEY,
  user_id integer, 
  email varchar(255),
  FOREIGN KEY (user_id) REFERENCES Users (user_id)
);

CREATE TABLE Image_Details (
  image_id integer PRIMARY KEY,
  user_id integer,
  name varchar(255),
  FOREIGN KEY (user_id) REFERENCES Users (user_id)
);

CREATE TABLE Image_Binary (
  id integer PRIMARY KEY,
  image_id integer,
  raw varbinary,
  {}
  FOREIGN KEY (image_id) REFERENCES Image_Details (image_id )
);

CREATE TABLE Blend (
  id integer PRIMARY KEY,
  image_id integer,
  source_image1 integer,
  source_image2 integer,
  blend_ratio float,
  FOREIGN KEY (image_id) REFERENCES Image_Details (image_id ),
  FOREIGN KEY (source_image1) REFERENCES Image_Details (image_id ),
  FOREIGN KEY (source_image2) REFERENCES Image_Details (image_id )
);


