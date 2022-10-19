---
layout: page
title: Research work
description: Research work of Tong Guo
permalink: /research/
---

<ul>

  <a class="page-link" href="/research/relabel/">relabel</a>

  {% for post in site.categories.research %}
    <li>
        <a href="{{ post.url }}" title="{{ post.title }}">{{ post.title }}</a>
    </li>
  {% endfor %}
</ul>
