const { withEnzyme } = require('jest-expo-enzyme');

module.exports = {
    projects: [
        // Skipping Node because we want to test DOM presets only
        withEnzyme(require('jest-expo/ios/jest-preset')),
        // withEnzyme(require('jest-expo/android/jest-preset')),
        // The Enzyme support added to web is different from that added to native, which `withEnzyme` handles
        // Luckily you won't have to do anything special because it reads the platform from
        // `haste.defaultPlatform` of the provided Jest config
        // withEnzyme(require('jest-expo/web/jest-preset')),
    ],
};