// Independent test expectation; do not import the production path helper.
export const fixtureRunPath = ({ id, source }) => `runs/${source.branch === 'develop' ? 'default-branch' : 'side-branches'}/${id}.json`;
