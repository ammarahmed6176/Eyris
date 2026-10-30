# TODO

- [x] Update pubspec.yaml to include assets/tiles/pakistan.mbtiles
- [x] Fix socket board hallucinations on door handles, laptops, sofa patterns, and picture frames
- [x] Improve door and armchair detection sensitivity across models (lower thresholds + add roomModel backedThresholds)
- [x] Add scene rule suppressing cushions/pillows on sofas from being reported as backpacks
- [x] Add scene rule preventing laptops from being misdetected as sofas
- [x] Release EYRIS 1.1.1 APK cleanly compiled with 52/52 passing tests
- [x] Accurate distance estimation for all objects (including small objects like socket board, bottle, and large objects like electric pole, truck, wall)
- [x] Clear directional voice alerts and on-screen cards (Left, Right, Front/Straight ahead)
- [x] Electric poles & utility poles direct detection and protection against false person classifications
- [x] Stairs detection and safeguard against false speed breaker misclassifications
- [x] Socket board detection on wall surfaces overriding generic wall detections
- [x] Unit tests updated and expanded for all objects, distances, and directional features
- [ ] Refactor OfflineMBTilesMapScreen state handling (separate mbtiles ready vs loading)
- [ ] Improve geolocator permission handling (deniedForever) + mounted checks
- [ ] Remove unused import and simplify map controller move timing


