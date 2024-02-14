import setuptools


with open('README.md', 'r') as f:
  long_description = f.read()

setuptools.setup(
  name='invisible-watermark-gpu',
  version='0.2.4',
  author='Will Drevo',
  author_email='will.drevo@stability.ai',
  description='GPU adaptation for the library for creating and decoding invisible image watermarks',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/Stability-AI/invisible-watermark-gpu',
  packages=setuptools.find_packages(),
  install_requires=[
      'opencv-python-headless==4.8.0.76',
      'torch>=2.0.0',
      'Pillow>=6.0.0',
      'PyWavelets>=1.1.1',
      'numpy>=1.17.0',
      'pycudwt-multitarget==1.0.5',
      'pytest',
  ],
  scripts=['invisible-watermark'],
  classifiers=[
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: MIT License',
      'Operating System :: POSIX :: Linux',
  ],
  # Python 3.6 tested in Ubuntu 18.04 LTS.
  python_requires='>=3.6',
  include_package_data=True,
)
