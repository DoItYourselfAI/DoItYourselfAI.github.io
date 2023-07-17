---
layout: front
permalink: /
title: "Home"
front_banner: "/images/front_banner.png"
---

<div class="tiles">
{% for post in site.posts %}
	{% include post-grid.html %}
{% endfor %}
</div>