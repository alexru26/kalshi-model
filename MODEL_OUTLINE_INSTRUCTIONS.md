# Instructions
For the questions below, please start by describing a simple "level zero" model and then a more complex "level one" (and "level two" if useful) model.

Please give us a detailed outline of these model. This outline should describe your model in enough detail so that anyone could read it and code your model using any relevant computer programming language.   

Your model outline should (at the very least) answer the next set of questions. One format you could use is to start each section by listing the numbered bullet point below, followed by your answers to each set of questions (as shown in the first question).

1. What is the specific purpose of your model? Are you trying to predict or understand a phenomenon? What is that phenomenon? What are the relevant stylized facts)

2. What are the main parts of your model? What are the main agent types in the model (people, institutions, etc.)?

3. For each type of agent in your model: What actions can this agent take? Does this agent have goals, if so, what are they?  What information is available to this agent and when is it available?   What governs this agent’s behavior?  How many such agents exist in your model (and what is the possible range in this number)?  Are the agents of this type homogeneous or heterogeneous?
4. What other relevant objects exist in that space?
5. What type of space do the agents exist in (for example, a two-dimensional plane, a network, etc.)?   What does space and time represent in your model? Does physical location matter or not? If it matters, what does it represent? What does time represent in your model? Are we thinking about a process that evolves over minutes, hours, days, or years?  Do the agents act synchronously or asynchronously?
6. How do the various parts of your model interact with one another? How often, and what happens when they do? What things change and how?  Be sure and describe this aspect in very clear detail.
7. How will your model be initialized? What is the starting state of the model?
8. Under what conditions will your models stop? Will it run until something happens? Or will it run until a fixed amount of time has passed? Or something else?
9. What are the parameters of your model that you will explore? What do they represent and what values will they take?
10. What things will you measure about your model?  What are the outputs of interest?

# Example for Schelling's model of segregation:

This model is to explore the ways that relatively mild preferences for not being a minority in a neighborhood might affect the amount of segregation in a given city.  We do not anticipate that our model will make precise predictions about segregation within an actual city, rather we are looking for the general patterns that might emerge in such systems.

The model is made up of two main parts: housing people and housing locations.  

People are the key agent in the model.  They are divided into two types which we label "red" and "green."  This type represents some kind of observable personal difference that is relevant to housing choices.  Although our inspiration was race and nationality, we think our model might also apply to other forms of segregation like political opinion or sexual orientation.  Importantly, the type of a person does not change over time. It is fixed and constant throughout the model.  We will treat these "people" as standing in for family units who move together.

Housing locations are on a grid, like a checkerboard.  Those locations may be either occupied or unoccupied. All housing locations are otherwise identical, as we are not trying to model differential housing cost or quality. Space in this model represents physical housing locations in a city.  We will experiment with different city sizes, but we anticipate it being moderately large, say around 1,000 housing locations.

Time in the model proceeds as follows:

On each time-step, we first ask all the people to decide if they are happy or unhappy.  People are unhappy if less than a percentage of their nearest eight neighbors have the same type (red or green) as that person.  This percentage is a parameter of the model called the tolerance parameter, which is the same for all people regardless of type.  A person is unhappy if the number of identical neighbors is below the tolerance parameter.

Next, one of the unhappy individuals will be randomly chosen and will move to a randomly chosen, unoccupied housing location.  (We might modify this so that multiple unhappy agents move, without any agents recalculating their happiness during such moves.)

The above time steps will be iterated, until there are no unhappy people or a specified number of timesteps have occurred (this latter condition will prevent situations where the model never stops).

The model will be initialized by creating some number of red and green people (this number is a parameter of the model) and placing them at random housing locations.

The parameters of interest in the model are the number of red people, the number of green people, and the tolerance parameter.

The outputs we will look at are: how similar is each person to their neighbors as the model iterates, and how many times steps does it take for the model to stabilize (if at all).
