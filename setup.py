import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(name='keras_webhook_callback',
      description='Receive notifications about your model training on your messaging app!',
      long_description=long_description,
      long_description_content_type="text/markdown",
      version='0.2.3',
      url='https://github.com/AlexBella365/keras_webhook_callback',
      author='Alex Bella',
      author_email='alexbella365@gmail.com',
      license='GNU General Public License v3 (GPLv3)',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3',
      ],
      keywords='api python keras training callback',
      install_requires=[
        'python-telegram-bot>=12',
        'matplotlib>=3',
        'keras>=2.2',
        'requests>=2',
        'humanize>=2'
      ],
      packages=setuptools.find_packages(),
      python_requires='>=3.6'
)
