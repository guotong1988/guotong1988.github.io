---
layout: page
title: research works by time order
description: Research papers of Tong Guo
permalink: /research/
---

<ul>

  {% for post in site.categories.research %}
    <li>
        <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}


</ul>
<meta name="google-site-verification" content="8NeXeopl0Y7RpgHgRilAMtTLuzHTNav3LpL8MA7lj1A" />
