<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/reevesba/protego">
    <img src="img/logo.png" alt="Logo" width="125px">
  </a>
  <h1 align="center">Protego</h1>
  <p align="center">
    Detect SQL Injection Payloads
    <br />
    <a href="https://github.com/reevesba/protego"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://protego-app.xyz" target="_blank">View Web Application</a>
    ·
    <a href="https://github.com/reevesba/protego/issues">Report Bug</a>
    ·
    <a href="https://github.com/reevesba/protego/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project
Protego essentially bundles batch modeling algorithms from SciKit Learn with online modeling algorithms from River. It's purpose is to build machine learning models capable of detecting SQL injection payloads. The only items in this package restricted to SQL injection is a DataLoader class, which will fetch 115k samples that I have collected, and the FeatureExtractor class. The models themselves can be trained for any type of classification task. 

### Built With
<a href="https://www.python.org/" target="_blank">
  <img align="left" width="32px" src="https://cdn.jsdelivr.net/npm/simple-icons@3.13.0/icons/python.svg" alt="python">
</a>
<a href="https://pandas.pydata.org/" target="_blank">
  <img align="left" width="32px" src="https://cdn.jsdelivr.net/npm/simple-icons@3.13.0/icons/pandas.svg" alt="pandas">
</a>
<a href="https://riverml.xyz/latest/" target="_blank">
  <img align="left" width="32px" src="img/river.png" alt="numpy">
</a>
<a href="https://scikit-learn.org/stable/" target="_blank">
  <img align="left" width="32px" src="https://cdn.jsdelivr.net/npm/simple-icons@3.13.0/icons/scikit-learn.svg" alt="sklearn">
</a>

<br />
<br />

<!-- GETTING STARTED -->
## Getting Started
This package can be downloaded using PyPI.

### Installation
   ```sh
   pip3 install protego
   ```

<!-- ROADMAP -->
## Roadmap
See the [open issues](https://github.com/reevesba/protego/issues) for a list of proposed features (and known issues).


<!-- CONTRIBUTING -->
## Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/newFeature`)
3. Commit your Changes (`git commit -m 'adding new feature xyz'`)
4. Push to the Branch (`git push origin feature/newFeature`)
5. Open a Pull Request

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Bradley Reeves - reevesbra@outlook.com

Project Link: [https://github.com/reevesba/protego](https://github.com/reevesba/protego)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/reevesba/protego.svg?style=for-the-badge
[contributors-url]: https://github.com/reevesba/protego/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/reevesba/protego.svg?style=for-the-badge
[forks-url]: https://github.com/reevesba/protego/network/members
[stars-shield]: https://img.shields.io/github/stars/reevesba/protego.svg?style=for-the-badge
[stars-url]: https://github.com/reevesba/protego/stargazers
[issues-shield]: https://img.shields.io/github/issues/reevesba/protego.svg?style=for-the-badge
[issues-url]: https://github.com/reevesba/protego/issues
[license-shield]: https://img.shields.io/github/license/reevesba/protego.svg?style=for-the-badge
[license-url]: https://github.com/reevesba/protego/blob/master/LICENSE.txt