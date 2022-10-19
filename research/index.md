---
layout: page
title: Research work
description: Research work of Tong Guo
permalink: /research/
---

<ul>
  {% for post in site.categories.research %}
    <li>
        <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
