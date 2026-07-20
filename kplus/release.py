
RELEASE_LEVELS = [ALPHA, BETA, RELEASE_CANDIDATE, FINAL] = ['alpha', 'beta', 'candidate', 'final']
RELEASE_LEVELS_DISPLAY = {ALPHA: 'a', BETA: 'b',
                          RELEASE_CANDIDATE: 'rc', FINAL: ''}
                          
# version_info format: (MAJOR, MINOR, MICRO, RELEASE_LEVEL, SERIAL)
version_info = (3, 0, 1, BETA, 0, '')
series = serie = major_version = '.'.join(str(s) for s in version_info[:2])

MIN_PY_VERSION = (3, 12)
MAX_PY_VERSION = (3, 14)

class Release:
    version: str = series + RELEASE_LEVELS_DISPLAY[version_info[3]] + str(version_info[4] or '') + version_info[5]
    product_name: str = "kplus"
    description: str = "Karaoke+ by iantirta"
    long_desc: str = """Karaoke+ is a full automation engine for generating karaoke with only youtube watch url as an input.
it will automatically fetch and process all the necessary including lyrics, separating audio track, multiplexing,
aligning the lyrics to the audio, and returning a full makeover karaoke video."""
    
    classifiers = """Development Status :: 4 - Beta
License :: OSI Approved :: MIT License

Programming Language :: Python
Operating System :: OS Independent
Intended Audience :: Developers
Intended Audience :: Science/Research
Topic :: Multimedia :: Sound/Audio
Topic :: Multimedia :: Sound/Audio :: Mixers
"""

    url = 'https://www.iantirta.com'
    author = 'iantirta.com'
    author_email = 'ian@iantirta.com'
    license = 'MIT'
    