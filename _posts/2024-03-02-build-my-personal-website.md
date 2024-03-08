---
title: Build my personal website
layout: post
post-image: "https:jizhaoqin.github.io/assets/images/post/website-demo.png"
description: Major processes to build a static personal website and deploy it with github pages.
tags:
- static website
- setup
---

# Build my personal website

## Template

- Nowadays website is much more complicated than before. So the most efficient way to create a personal website is to find a template that meets the requirements.
- For me a static website is enough, you can find the appropriate website template in [`http://jekyllthemes.org`](http://jekyllthemes.org/). I'll use [`WhatATheme`](http://jekyllthemes.org/themes/what-a-theme/) as the template of my website.

- A brief intro video WhatATheme:

<iframe width="560" height="315" src="https://www.youtube.com/embed/VfPa2c9kwhQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## Deployment

- After deciding which template to use for the website, we can fork the repository on [github](https://github.com/thedevslot/WhatATheme).
- then we need to change the repo name to `<username>.github.io`, supposed to be `jizhaoqin.github.io`.
- then we could use the `Actions` feature provided by `github pages` to deploy our website. The website would be rebuild automatically after every time the repository is refreshed. And everybody can visit the site `<username.github.io>`, for me again is [`jizhaoqin.github.io`](https://jizhaoqin.github.io).
- Of course we could deploy the website with our own server instead of github server, for example use AWS cloud server or other cloud service. But for simple demand like a small personal website without a big amount of visits, `github pages` is sufficient and convenient.

---

## Customize

After successful deployment of the website, the next step is to customize the contents and layout according to our need and preference.

### Home Page

The home page includes the following sections:

- A Hero section - A section where you can outsource an image which will work as the background for the particular section; it also will include your name and a tagline which can be easily manipulated via the _config.yml file.
- An About section - A section where you can include your image and a 60 word paragraph which again you can easily manipulate using the _config.yml file.
- A Contact section - A section where you can include 3 direct ways to contact
  - `Ping on Messenger`
  - `Send an Email`
  - `Tweet on Twitter`
  - The contact section will also include 10 different social media buttons for your audience to follow.
`Facebook`, `Twitter`, `Instagram`, `LinkedIn`, `GitHub`, `YouTube`, `Reddit`, `Behance`, `Dribbble` & `Spotify`.
s
- > of course we do not need all the social media buttons, we can easily remove the unwanted buttons by editing the in the `_include/contact.html` file or by delete the corresponding attributes in `_config.yml`.

### Project and Blog

- The project or blog site includes a horizontal card list where the latest articles are fetched from the `_posts` folder ***automatically***.
- It also includes an instant search box which matches your query from the title, description & content of your post and shows the result as soon as you type.
- The blog card will include:
  - Post Title
  - a specific amount of words from the content of the post
  - The publish date
  - The time which will be required to read the post.

- > if we want to add new project card to the site, we can add files with name `yyyy-mm-dd-title.md` into the `_posts` folder, and the server would find it and automatically generates necessary files and deploys the correspond site.

### Footer

- The footer includes
  - `copyright` and `Name of the Author`
  - `credits`

- > other contents can also be added if needed.

---

## Extra Features of `WhatATheme`

WhatATheme comes pre installed with

- **`HTML Compressor`** - It'll compress all the pages by removing any extra space or blank lines.
- **`Google Analytics`** - A web analytics service offered by Google that tracks and reports website traffic. For more information [click here](https://analytics.google.com){:target="blank"}.
- **`Disqus`** - A worldwide blog comment hosting service for web sites and online communities that use a networked platform. For more information about Disqus [click here](https://help.disqus.com/en/articles/1717053-what-is-disqus){:target="blank"}

- > For more information about `WhatATheme` [click here](https://github.com/thedevslot/WhatATheme/blob/gh-pages/README.md){:target="blank"}

---
