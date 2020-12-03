import React from 'react';
import {createStackNavigator} from "@react-navigation/stack";
import {NavigationContainer} from "@react-navigation/native";
import Demo from "./components/Demo";
import Home from "./components/Home";

const Stack = createStackNavigator();

/**
 * contains the navigation component
 * @constructor
 */
export default function App() {
  return (
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen
              name='Home'
              component={Home}
              options={{headerShown: false}}
          />
          <Stack.Screen
              name='image'
              component={Demo}
          />
        </Stack.Navigator>
      </NavigationContainer>
  );
}